use tch::{nn, IndexOp, Kind, Tensor};
use tch::nn::Module;

pub const IMG_SIZE: i64 = 518;
const PATCH_SIZE: i64 = 14;
const NUM_CLASSES: i64 = 1000;

#[derive(Debug)]
struct Attention {
    qkv: nn::Linear,
    output: nn::Linear,
    num_heads: i64,
    scale: f64,
}

impl Attention {
    fn new(
        vs: nn::Path,
        dim: i64,
        num_heads: i64,
        qkv_bias: bool,
        proj_bias: bool,
    ) -> Self {
        let qkv_config = nn::LinearConfig { bias: qkv_bias, ..Default::default() };
        let proj_config = nn::LinearConfig { bias: proj_bias, ..Default::default() };

        let attn = &vs / "attention";
        let query = nn::linear(&attn / "query", dim, dim, qkv_config);
        let key = nn::linear(&attn / "key", dim, dim, qkv_config);
        let value = nn::linear(&attn / "value", dim, dim, qkv_config);

        let qkv_weight = Tensor::cat(&[query.ws, key.ws, value.ws], 0);
        let qkv_bias = if qkv_bias {
            Some(Tensor::cat(
                &[
                    query.bs.unwrap(),
                    key.bs.unwrap(),
                    value.bs.unwrap(),
                ],
                0,
            ))
        } else {
            None
        };

        let qkv = nn::Linear { ws: qkv_weight, bs: qkv_bias };
        let output = nn::linear(&vs / "output" / "dense", dim, dim, proj_config);
        let scale = 1. / ((dim / num_heads) as f64).sqrt();
        Self {
            qkv,
            output,
            num_heads,
            scale,
        }
    }
}

impl nn::Module for Attention {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let (b, n, c) = xs.size3().unwrap();
        let qkv = self
            .qkv
            .forward(xs)
            .reshape([b, n, 3, self.num_heads, c / self.num_heads])
            .permute([2, 0, 3, 1, 4]);
        let q = qkv.get(0) * self.scale;
        let k = qkv.get(1);
        let v = qkv.get(2);
        let attn = q.matmul(&k.transpose(-2, -1)).softmax(-1, Kind::Float);
        attn.matmul(&v).transpose(1, 2).reshape([b, n, c]).apply(&self.output)
    }
}

#[derive(Debug)]
struct LayerScale {
    lambda1: Tensor,
}

impl LayerScale {
    fn new(vs: nn::Path, dim: i64) -> Self {
        let lambda1 = vs.var("lambda1", &[dim], nn::Init::Const(0.));
        Self { lambda1 }
    }
}

impl nn::Module for LayerScale {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs * &self.lambda1
    }
}

#[derive(Debug)]
struct Mlp {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl Mlp {
    fn new(vs: nn::Path, in_features: i64, hidden_features: i64, bias: bool) -> Self {
        let out_features = in_features;
        let config = nn::LinearConfig { bias, ..Default::default() };
        let fc1 = nn::linear(&vs / "fc1", in_features, hidden_features, config);
        let fc2 = nn::linear(&vs / "fc2", hidden_features, out_features, config);
        Self { fc1, fc2 }
    }
}

impl nn::Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.fc1).gelu("none").apply(&self.fc2)
    }
}

#[derive(Debug)]
struct Block {
    norm1: nn::LayerNorm,
    attn: Attention,
    ls1: LayerScale,
    norm2: nn::LayerNorm,
    mlp: Mlp,
    ls2: LayerScale,
}

impl Block {
    fn new(vs: nn::Path, dim: i64, num_heads: i64) -> Self {
        let norm1 = nn::layer_norm(&vs / "norm1", vec![dim], Default::default());
        let attn = Attention::new(&vs / "attention", dim, num_heads, true, true);
        let ls1 = LayerScale::new(&vs / "layer_scale1", dim);
        let norm2 = nn::layer_norm(&vs / "norm2", vec![dim], Default::default());
        let mlp = Mlp::new(&vs / "mlp", dim, dim * 4, true);
        let ls2 = LayerScale::new(&vs / "layer_scale2", dim);
        Self {
            norm1,
            attn,
            ls1,
            norm2,
            mlp,
            ls2,
        }
    }
}

impl nn::Module for Block {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let residual = xs;
        let xs = self
            .ls1
            .forward(&self.attn.forward(&self.norm1.forward(xs)));
        let xs = xs + residual;
        let residual = &xs;
        let xs = self
            .ls2
            .forward(&self.mlp.forward(&self.norm2.forward(&xs)));
        xs + residual
    }
}

#[derive(Debug)]
struct PatchEmbed {
    proj: nn::Conv2D,
    patch_size: (i64, i64),
    num_patches: i64,
}

impl PatchEmbed {
    fn new(
        vs: nn::Path,
        img_size: i64,
        patch_size: i64,
        in_chans: i64,
        embed_dim: i64,
    ) -> Self {
        let config = nn::ConvConfig { stride: patch_size, ..Default::default() };
        let proj = nn::conv2d(&vs / "projection", in_chans, embed_dim, patch_size, config);
        let num_patches = (img_size / patch_size) * (img_size / patch_size);
        Self {
            proj,
            patch_size: (patch_size, patch_size),
            num_patches,
        }
    }
}

impl nn::Module for PatchEmbed {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let (_b, _c, h, w) = xs.size4().unwrap();
        let (patch_h, patch_w) = self.patch_size;
        if (h % patch_h) != 0 {
            panic!("image height {h} is not a multiple of patch height {patch_h}")
        }
        if (w % patch_w) != 0 {
            panic!("image width {w} is not a multiple of patch width {patch_w}")
        }
        let xs = self.proj.forward(xs);
        let (b, c, h, w) = xs.size4().unwrap();
        // flatten embeddings.
        xs.reshape([b, c, h * w]).transpose(1, 2)
    }
}

#[derive(Debug)]
pub struct DinoVisionTransformer {
    patch_embed: PatchEmbed,
    cls_token: Tensor,
    pos_embed: Tensor,
    blocks: Vec<Block>,
    norm: nn::LayerNorm,
    head: Option<nn::Linear>,
}

impl DinoVisionTransformer {
    pub fn new(
        vs: nn::Path,
        vs_head: Option<nn::Path>,
        depth: i64,
        embed_dim: i64,
        num_heads: i64,
    ) -> Self {
        let vs_embeddings = &vs / "embeddings";
        let patch_embed = PatchEmbed::new(
            &vs_embeddings / "patch_embeddings",
            IMG_SIZE,
            PATCH_SIZE,
            3,
            embed_dim,
        );
        let cls_token = vs_embeddings.var("cls_token", &[1, 1, embed_dim], nn::Init::Const(0.));
        let num_tokens = 1;
        let pos_embed = vs_embeddings.var(
            "position_embeddings",
            &[1, patch_embed.num_patches + num_tokens, embed_dim],
            nn::Init::Const(0.),
        );
        let head = match vs_head {
            Some(vs_head) => Some(nn::linear(vs_head, 2 * embed_dim, NUM_CLASSES, Default::default())),
            None => None,
        };
        let norm = nn::layer_norm(&vs / "layernorm", vec![embed_dim], Default::default());
        let vs_layer = &vs / "encoder" / "layer";
        let blocks = (0..depth)
            .map(|i| Block::new(&vs_layer / i.to_string(), embed_dim, num_heads))
            .collect();
        Self {
            patch_embed,
            cls_token,
            pos_embed,
            blocks,
            norm,
            head,
        }
    }

    fn interpolate_pos_encoding(&self, xs: &Tensor, w: i64, h: i64) -> Tensor {
        let npatch = xs.size()[1] - 1;
        let n = self.pos_embed.size()[1] - 1;
        let sqrt_n = (n as f64).sqrt();
        if npatch == n && w == h {
            return xs.shallow_clone();
        }
        let class_pos_embed = self.pos_embed.i((.., ..1));
        let patch_pos_embed = self.pos_embed.i((.., 1..));
        let dim = *xs.size().last().unwrap();
        let (w0, h0) = ((w / PATCH_SIZE) as f64 + 0.1, (h / PATCH_SIZE) as f64 + 0.1);
        let patch_pos_embed = patch_pos_embed
            .reshape([1, sqrt_n as i64, sqrt_n as i64, dim])
            .permute([0, 3, 1, 2])
            .upsample_bicubic2d([w0 as i64, h0 as i64], false, w0 / sqrt_n, h0 / sqrt_n)
            .permute([0, 2, 3, 1])
            .reshape([1, -1, dim]);
        Tensor::cat(&[&class_pos_embed, &patch_pos_embed], 1)
    }

    fn prepare_tokens_with_mask(&self, xs: &Tensor) -> Tensor {
        let (_b, _nc, w, h) = xs.size4().unwrap();
        let xs = self.patch_embed.forward(xs);
        let xs = Tensor::cat(&[&self.cls_token, &xs], 1);
        &xs + &self.interpolate_pos_encoding(&xs, w, h)
    }
}

impl nn::Module for DinoVisionTransformer {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut xs = self.prepare_tokens_with_mask(xs);
        for blk in self.blocks.iter() {
            xs = blk.forward(&xs);
        }
        let xs = self.norm.forward(&xs);
        let xs_norm_clstoken = xs.i((.., 0));
        let xs_norm_patchtokens = xs.i((.., 1..)).mean_dim(1, false, None);
        let xs = Tensor::concat(&[xs_norm_clstoken, xs_norm_patchtokens], -1);

        match &self.head {
            Some(head) => head.forward(&xs),
            None => xs,
        }
    }
}

pub fn vit_base(vs: nn::Path, vs_head: Option<nn::Path>) -> DinoVisionTransformer {
    DinoVisionTransformer::new(vs, vs_head, 12, 768, 12)
}
