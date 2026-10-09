# Survey on World Models

## TL;DR
- World models are internal simulators that learn and represent the structure and dynamics of environments, enabling agents to predict future states and plan actions effectively [1][2].
- Architecturally, world models range from observation-level generative models to latent-space dynamics and structured 3D representations, with modern approaches leveraging neural networks, transformers, and hybrid systems [1][3][4].
- Applications of world models are prominent in AI and robotics, improving decision-making, planning, and learning efficiency by simulating environment dynamics and reducing real-world trial costs [3][5][6][7].
- Key challenges include scalability for long-horizon reasoning, generalization beyond training data, interpretability of diverse representations, and integration with action conditioning and safety mechanisms [2][8].
- Future research directions emphasize unified frameworks, richer multimodal data, sim-to-real adaptation, safety, and shifting focus from visual realism to functional utility [2][8].

## Background
World models have their conceptual roots in cognitive science, where Craik (1943) proposed that organisms maintain internal "working models" of the world to simulate and predict outcomes without direct trial and error. In artificial intelligence, world models formalize this idea as internal simulators that learn environment dynamics from sensory data, enabling agents to plan and act efficiently. They are mathematically framed within partially observable Markov decision processes (POMDPs) and operate by compressing multimodal sensory inputs into stable latent representations that support prediction and planning [1][2].

## Definitions and Fundamental Concepts
World models serve dual purposes: understanding the present environment by compressing sensory data into meaningful representations, and predicting future states to enable planning. They are functionally categorized into renderers (generating perceptual outputs), simulators (predicting state transitions), and planners (selecting actions based on predicted futures). Unlike large language models, world models operate on sensor data and support causal reasoning and action-conditioned predictions [1][2].

Mathematically, world models can be viewed as homomorphic images of the data generation process, with internal representations that causally determine network outputs, enabling control and interpretability. Emergent world models arise implicitly within neural networks trained on related tasks [1].

## Architectures of World Models
World models exhibit diverse architectures:
- Observation-level generative models predict in perceptual spaces (e.g., pixels or video frames).
- Latent-space dynamics models encode compact state representations and predict future embeddings, exemplified by architectures like Dreamer and JEPA (joint-embedding predictive architecture).
- Structured 3D or spatial models generate persistent environments with explicit geometry and topology, supporting long-term consistency and physical reasoning.

Hybrid models integrate predictive state modeling with action generation, often using multimodal inputs and self-supervised learning. Examples include World Action Models and multimodal transformer-based systems like Cosmos 3 [1][3][4].

## Applications of World Models
World models are extensively applied in AI and robotics to enhance decision-making, planning, and learning. They enable agents to anticipate future states, perform imagination-driven planning through simulated rollouts, and learn efficiently with reduced real-world interaction. Applications include robotic manipulation, autonomous driving, video analytics, and physical AI platforms. Advances in video generation models and foundation-scale platforms have further expanded their utility [3][5][6][7][4].

## Challenges and Limitations
Despite progress, world models face significant challenges:
- Scalability and long-horizon reasoning remain difficult due to error accumulation and instability over extended predictions.
- Generalization beyond training environments is limited by data scarcity and the simulation-to-reality gap.
- Interpretability varies with representation types, and some models prioritize visual realism over physical and causal consistency.
- Integration with action conditioning and safety mechanisms is underdeveloped, posing risks in real-world deployment.
- Benchmarking lacks unified metrics emphasizing functional utility over perceptual quality [8][2].

## Trends and open problems
Future research in world models is poised to address these challenges by:
- Developing unified definitions and frameworks that encompass both understanding and prediction.
- Improving long-horizon temporal coherence and action-conditioned modeling.
- Collecting richer, multimodal datasets including real-world interactions.
- Bridging the simulation-to-reality gap for better generalization.
- Enhancing safety, uncertainty estimation, and robustness for reliable deployment.
- Leveraging multimodal large language models to augment reasoning and representation.
- Shifting focus from visual realism to functional utility to better support downstream tasks and decision-making.

This survey synthesizes recent advances and challenges in world model research, highlighting their critical role in advancing AI and robotics toward more intelligent, anticipatory, and efficient systems.

## References
[1] A Definition and Roadmap for World Models. arxiv. https://arxiv.org/abs/2607.06401 (2026-07-01)
[2] Understanding World or Predicting Future? A Comprehensive Survey of World Models. web. https://dl.acm.org/doi/10.1145/3746449 (2025-01-01)
[3] World Model for Robot Learning: A Comprehensive Survey. arxiv. https://arxiv.org/abs/2605.00080 (2026-05-01)
[4] What Is a World Model? | NVIDIA Glossary. web. https://www.nvidia.com/en-us/glossary/world-models/ (n.d.)
[5] Video Generation Models in Robotics -- Applications, Research Challenges, Future Directions. hf-search. https://huggingface.co/papers/2601.07823 (2026-01-12)
[6] DayDreamer: World Models for Physical Robot Learning. hf-search. https://huggingface.co/papers/2206.14176 (2022-06-28)
[7] Cosmos World Foundation Model Platform for Physical AI. hf-search. https://huggingface.co/papers/2501.03575 (2025-01-07)
[8] State of World Models 2026: Taxonomy, Benchmarks and Open Challenges. web. https://world-models.io/reports/state-of-world-models-2026/state-of-world-models-2026-v1.0.pdf (2026-01-01)
