// This is an adaptation of the following code to the model available on Hugging Face
// as facebook/dinov2-with-registers-base:
// https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/dinov2reg4.rs

//! Implementation of the DINOv2 revision (4 registers)
//!
//! The DINOv2-reg4 model is a variant of DINOv2 that adds 4 register tokens to the
//! original architecture.
//!
//! - [Paper](https://arxiv.org/abs/2309.16588). DINOv2: Learning Robust Visual Features without Supervision
//! - [GH Repo](https://github.com/facebookresearch/dinov2)

use candle_core::{IndexOp, Result, Tensor, D};
use candle_nn::{layer_norm, LayerNorm, linear_b, Linear, Module, VarBuilder};

const IMG_SIZE: usize = 518;
const PATCH_SIZE: usize = 14;

#[derive(Debug)]
struct Attention {
    qkv: Linear,
    output: Linear,
    num_heads: usize,
    scale: f64,
}

impl Attention {
    fn new(
        vb: VarBuilder,
        dim: usize,
        num_heads: usize,
        qkv_bias: bool,
        proj_bias: bool,
    ) -> Result<Self> {
        let attn = vb.pp("attention");
        let query = linear_b(dim, dim, qkv_bias, attn.pp("query"))?;
        let key = linear_b(dim, dim, qkv_bias, attn.pp("key"))?;
        let value = linear_b(dim, dim, qkv_bias, attn.pp("value"))?;
        let qkv_weight = Tensor::cat(&[query.weight(), key.weight(), value.weight()], 0)?;
        let qkv_bias = if qkv_bias {
            Some(Tensor::cat(
                &[
                    query.bias().unwrap(),
                    key.bias().unwrap(),
                    value.bias().unwrap(),
                ],
                0,
            )?)
        } else {
            None
        };
        let qkv = Linear::new(qkv_weight, qkv_bias);
        let output = linear_b(dim, dim, proj_bias, vb.pp("output").pp("dense"))?;
        let scale = 1. / ((dim / num_heads) as f64).sqrt();
        Ok(Self {
            qkv,
            output,
            num_heads,
            scale,
        })
    }
}

impl Module for Attention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, n, c) = xs.dims3()?;
        let qkv = self
            .qkv
            .forward(xs)?
            .reshape((b, n, 3, self.num_heads, c / self.num_heads))?
            .transpose(1, 2)? // 02134
            .transpose(0, 1)? // 20134
            .transpose(2, 3)?; // 20314
        let q = (qkv.i(0)? * self.scale)?;
        let k = qkv.i(1)?.contiguous()?;
        let v = qkv.i(2)?.contiguous()?;
        let attn = candle_nn::ops::softmax(&q.matmul(&k.t()?)?, D::Minus1)?;
        let attn = attn.matmul(&v)?.transpose(1, 2)?.reshape((b, n, c))?;
        self.output.forward(&attn)
    }
}

#[derive(Debug)]
struct LayerScale {
    lambda1: Tensor,
}

impl LayerScale {
    fn new(vb: VarBuilder, dim: usize) -> Result<Self> {
        let lambda1 = vb.get(dim, "lambda1")?;
        Ok(Self { lambda1 })
    }
}

impl Module for LayerScale {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&self.lambda1)
    }
}

#[derive(Debug)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn new(vb: VarBuilder, in_features: usize, hidden_features: usize, bias: bool) -> Result<Self> {
        let out_features = in_features;
        let fc1 = linear_b(in_features, hidden_features, bias, vb.pp("fc1"))?;
        let fc2 = linear_b(hidden_features, out_features, bias, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?.gelu()?;
        self.fc2.forward(&xs)
    }
}

#[derive(Debug)]
struct Block {
    norm1: LayerNorm,
    attn: Attention,
    ls1: LayerScale,
    norm2: LayerNorm,
    mlp: Mlp,
    ls2: LayerScale,
}

impl Block {
    fn new(vb: VarBuilder, dim: usize, num_heads: usize) -> Result<Self> {
        let norm1 = layer_norm(dim, 1e-6, vb.pp("norm1"))?;
        let attn = Attention::new(vb.pp("attention"), dim, num_heads, true, true)?;
        let ls1 = LayerScale::new(vb.pp("layer_scale1"), dim)?;
        let norm2 = layer_norm(dim, 1e-6, vb.pp("norm2"))?;
        let mlp = Mlp::new(vb.pp("mlp"), dim, dim * 4, true)?;
        let ls2 = LayerScale::new(vb.pp("layer_scale2"), dim)?;
        Ok(Self {
            norm1,
            attn,
            ls1,
            norm2,
            mlp,
            ls2,
        })
    }
}

impl Module for Block {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self
            .ls1
            .forward(&self.attn.forward(&self.norm1.forward(xs)?)?)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .ls2
            .forward(&self.mlp.forward(&self.norm2.forward(&xs)?)?)?;
        xs + residual
    }
}

#[derive(Debug)]
struct PatchEmbed {
    proj: candle_nn::Conv2d,
    patch_size: (usize, usize),
    num_patches: usize,
}

impl PatchEmbed {
    fn new(
        vb: VarBuilder,
        img_size: usize,
        patch_size: usize,
        in_chans: usize,
        embed_dim: usize,
    ) -> Result<Self> {
        let config = candle_nn::Conv2dConfig {
            stride: patch_size,
            ..Default::default()
        };
        let proj = candle_nn::conv2d(in_chans, embed_dim, patch_size, config, vb.pp("projection"))?;
        let num_patches = (img_size / patch_size) * (img_size / patch_size);
        Ok(Self {
            proj,
            patch_size: (patch_size, patch_size),
            num_patches,
        })
    }
}

impl Module for PatchEmbed {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_b, _c, h, w) = xs.dims4()?;
        let (patch_h, patch_w) = self.patch_size;
        if (h % patch_h) != 0 {
            candle_core::bail!("image height {h} is not a multiple of patch height {patch_h}")
        }
        if (w % patch_w) != 0 {
            candle_core::bail!("image width {w} is not a multiple of patch width {patch_w}")
        }
        let xs = self.proj.forward(xs)?;
        let (b, c, h, w) = xs.dims4()?;
        // flatten embeddings.
        xs.reshape((b, c, h * w))?.transpose(1, 2)
    }
}

#[derive(Debug)]
pub struct DinoVisionTransformer {
    patch_embed: PatchEmbed,
    cls_token: Tensor,
    reg_token: Tensor,
    pos_embed: Tensor,
    blocks: Vec<Block>,
    norm: LayerNorm,
}

impl DinoVisionTransformer {
    pub fn new(vb: VarBuilder, depth: usize, embed_dim: usize, num_heads: usize) -> Result<Self> {
        let vbe = vb.pp("embeddings");
        let patch_embed =
            PatchEmbed::new(vbe.pp("patch_embeddings"), IMG_SIZE, PATCH_SIZE, 3, embed_dim)?;
        let cls_token = vbe.get((1, 1, embed_dim), "cls_token")?;
        let reg_token = vbe.get((1, 4, embed_dim), "register_tokens")?;
        let num_tokens = 1;
        let pos_embed = vbe.get((1, patch_embed.num_patches + num_tokens, embed_dim), "position_embeddings")?;
        let norm = layer_norm(embed_dim, 1e-6, vb.pp("layernorm"))?;
        let vb_b = vb.pp("encoder.layer");
        let blocks = (0..depth)
            .map(|i| Block::new(vb_b.pp(i.to_string()), embed_dim, num_heads))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch_embed,
            cls_token,
            reg_token,
            pos_embed,
            blocks,
            norm,
        })
    }

    fn interpolate_pos_encoding(&self, xs: &Tensor, w: usize, h: usize) -> Result<Tensor> {
        let npatch = xs.dim(1)? - 1;
        let n = self.pos_embed.dim(1)? - 1;
        let sqrt_n = (n as f64).sqrt();
        if npatch == n && w == h {
            return Ok(self.pos_embed.clone());
        }
        let patch_pos_embed = self.pos_embed.i((.., 1..))?;
        let dim = xs.dim(D::Minus1)?;
        let (w0, h0) = ((w / PATCH_SIZE) as f64 + 0.1, (h / PATCH_SIZE) as f64 + 0.1);
        let patch_pos_embed = patch_pos_embed
            .reshape((1, sqrt_n as usize, sqrt_n as usize, dim))?
            .transpose(2, 3)?
            .transpose(1, 2)?;
        // This uses bicubic interpolation in the original implementation.
        let patch_pos_embed = patch_pos_embed.upsample_nearest2d(h0 as usize, w0 as usize)?;
        let el_count = patch_pos_embed.shape().elem_count();
        patch_pos_embed
            .transpose(1, 2)?
            .transpose(2, 3)?
            .reshape((1, el_count / dim, dim))
    }

    fn prepare_tokens_with_mask(&self, xs: &Tensor) -> Result<Tensor> {
        let (_b, _nc, w, h) = xs.dims4()?;
        if (w != IMG_SIZE) || (h != IMG_SIZE) {
            panic!("Error: The input tensor should have the shape: Bx3x518x518.");
        }
        let xs = self.patch_embed.forward(xs)?;
        let xs = (&xs + &self.interpolate_pos_encoding(&xs, w, h)?)?;
        let xs = Tensor::cat(&[&self.cls_token, &self.reg_token, &xs], 1)?;
        Ok(xs)
    }
}

impl Module for DinoVisionTransformer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = self.prepare_tokens_with_mask(xs)?;
        for blk in self.blocks.iter() {
            xs = blk.forward(&xs)?
        }
        let xs = self.norm.forward(&xs)?;
        Ok(xs.i((.., 0))?)
    }
}

pub fn vit_base(vb: VarBuilder) -> Result<DinoVisionTransformer> {
    DinoVisionTransformer::new(vb, 12, 768, 12)
}
