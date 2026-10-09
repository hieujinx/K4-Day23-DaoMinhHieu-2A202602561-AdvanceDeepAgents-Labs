# Survey on Video and Multimodal Generation

## TL;DR
- Recent video generation methods have transitioned from GANs and VAEs to diffusion models and transformer-based architectures, which provide better training stability, scalability, and video quality [1][2][3][4].
- Multimodal generation techniques integrate video with text, audio, and images using unified architectures such as diffusion transformers, flow-matching frameworks, and dual-stream transformers, enabling synchronized and controllable generation [5][6][7].
- Evaluation of video and multimodal generation employs multi-dimensional benchmarks combining traditional metrics (IS, FID, FVD), embedding-based scores (CLIP, BLIP), and large multimodal language models, though challenges remain in cross-modal evaluation and computational complexity [8][9][10][11].

## Background
Video and multimodal generation have seen rapid advances driven by deep generative models. Early video generation relied on GANs and VAEs, but recent progress favors diffusion models and transformers for their superior capabilities. Multimodal generation extends video synthesis by incorporating other modalities such as text and audio, enabling richer and more controllable content creation. Evaluating these models requires comprehensive metrics that capture quality, alignment, and temporal coherence.

## State-of-the-Art Methods for Video Generation
Video generation has evolved through several paradigms:

- **GAN-based methods**: Early approaches like MoCoGAN and StyleGAN-V adapted image GANs to video, achieving high resolution but facing training instability and scalability issues [1].
- **VAE-based methods**: Video VAEs such as CV-VAE compress spatio-temporal information into latent spaces, offering stable training but sometimes lower fidelity [4].
- **Diffusion models**: Emerging as dominant, diffusion models use latent diffusion and transformer backbones to generate high-quality videos with better mode coverage, though inference is computationally intensive [2][3].
- **Transformer-based approaches**: Autoregressive and diffusion transformers model video tokens for scalable, high-fidelity generation, exemplified by models like Latte and FrameDiT which introduce efficient attention mechanisms [2][3].

Challenges include maintaining temporal coherence, balancing quality and efficiency, and scaling to complex datasets.

## Multimodal Generation Techniques
Recent multimodal generation models combine video with text, audio, and images using unified architectures:

- **Diffusion transformers**: Models like 3MDiT jointly model video, audio, and text streams with omni-blocks for feature fusion and dynamic conditioning [6].
- **Flow-matching frameworks**: MM-Sonate enables controllable audio-video generation with zero-shot voice cloning, using modality-specific encoders and shared embedding spaces [5].
- **Dual-stream transformers**: LTX-2 employs asymmetric dual-stream transformers with modality-specific VAEs and cross-attention for tight audiovisual alignment [7].

These architectures support diverse applications including speech synthesis, music generation, and audiovisual content creation.

## Evaluation Metrics and Benchmarks
Evaluating video and multimodal generation involves multiple dimensions:

- **Video quality**: Metrics like Inception Score (IS), Frechet Inception Distance (FID), and Frechet Video Distance (FVD) assess image quality, diversity, and temporal coherence.
- **Alignment and compositionality**: Embedding-based scores (CLIP, BLIP) and multimodal large language models evaluate semantic alignment with input conditions and compositional understanding.
- **Benchmarks**: Video-Bench, T2V-CompBench, VBench++, and MSVBench provide comprehensive, human-aligned evaluation frameworks covering quality, alignment, temporal consistency, and motion realism [8][9][10][11].

Limitations include embedding misalignment with human preferences, computational cost, and the complexity of cross-modal evaluation.

## Trends and open problems
- **Computational efficiency**: Diffusion models offer quality gains but require costly inference; efficient architectures and sampling methods remain a priority.
- **Temporal coherence and consistency**: Ensuring smooth, realistic motion across frames is challenging, especially in long videos.
- **Multimodal alignment**: Achieving tight synchronization and semantic consistency across modalities is an ongoing research focus.
- **Evaluation challenges**: Developing metrics that accurately reflect human judgment across modalities and compositional aspects is critical.
- **Scalability and generalization**: Models must handle diverse, complex datasets and generalize to open-domain scenarios.

Continued research in these areas will drive the next generation of video and multimodal generative models.

## References
[1] Evolution of Video Generative Foundations. arxiv. https://arxiv.org/abs/2604.06339 (2026-04-01)
[2] Latte: Latent Diffusion Transformer for Video Generation. arxiv. https://arxiv.org/abs/2401.03048 (2024-01-01)
[3] FrameDiT: Diffusion Transformer with Matrix Attention for Efficient Video Generation. web. https://openaccess.thecvf.com/content/CVPR2026F/papers/Le_FrameDiT_Diffusion_Transformer_with_Matrix_Attention_for_Efficient_Video_Generation_CVPRF_2026_paper.pdf (2026-06-01)
[4] CV-VAE: A Compatible Video VAE for Latent Generative Video Models. hf-daily. https://huggingface.co/papers/2405.20279 (2024-05-01)
[5] MM-Sonate: Multimodal Controllable Audio-Video Generation with Zero-Shot Voice Cloning. arxiv. https://arxiv.org/abs/2601.01568 (2026-01-01)
[6] 3MDiT: Unified Tri-Modal Diffusion Transformer for Text-Driven Synchronized Audio-Video Generation. arxiv. https://arxiv.org/abs/2511.21780 (2025-11-26)
[7] LTX-2: Efficient Joint Audio-Visual Foundation Model. arxiv. https://arxiv.org/abs/2601.03233 (2026-01-01)
[8] Video-Bench: Human-Aligned Video Generation Benchmark. web. https://openaccess.thecvf.com/content/CVPR2025/papers/Han_Video-Bench_Human-Aligned_Video_Generation_Benchmark_CVPR_2025_paper.pdf (2025-06-01)
[9] T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generation. web. https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_T2V-CompBench_A_Comprehensive_Benchmark_for_Compositional_Text-to-video_Generation_CVPR_2025_paper.pdf (2025-06-01)
[10] VBench++: Comprehensive and Versatile Benchmark Suite for Video Generative Models. arxiv. https://arxiv.org/abs/2411.13503 (2024-11-01)
[11] MSVBench: Towards Human-Level Evaluation of Multi-Shot Video Generation. web. https://aclanthology.org/2026.findings-acl.1203.pdf (2026-01-01)
