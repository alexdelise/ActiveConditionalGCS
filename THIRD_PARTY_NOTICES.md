# Third-Party Notices

## Stable Diffusion v1.5 Model Weights

This package uses Stable Diffusion v1.5 through Hugging Face Diffusers. The
model weights are not included in this repository; they are downloaded or loaded
from a local Hugging Face cache at runtime.

- Model ID used by the code:
  [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- Model page:
  [Hugging Face model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- Developed by: Robin Rombach, Patrick Esser, and contributors
- Copyright: Copyright (c) 2022 Robin Rombach and Patrick Esser and contributors
- License: CreativeML Open RAIL-M, dated August 22, 2022

Use of the Stable Diffusion v1.5 model weights is governed by the CreativeML
Open RAIL-M license and the model card. The license permits open and responsible
downstream use subject to its terms, including use-based restrictions that also
apply to derivatives of the model. Users who download or cache the weights for
these experiments are responsible for complying with that license.

## CS4ML

Portions of the K-tilde/Christoffel sampling implementation are adapted from:

- Repository: [JMcardenas/CS4ML](https://github.com/JMcardenas/CS4ML)
- Folder:
  [MRI-Generative-Models](https://github.com/JMcardenas/CS4ML/tree/main/MRI-Generative-Models)
- Upstream file of interest:
  [generative_cs_example.py](https://github.com/JMcardenas/CS4ML/blob/main/MRI-Generative-Models/generative_cs_example.py)
- URL: [GitHub repository](https://github.com/JMcardenas/CS4ML)
- License: MIT License

The adapted pieces include the Christoffel/K-tilde Monte Carlo estimator, the
partial Fourier helper convention, and the surrounding reconstruction workflow
needed to evaluate those sampling distributions in the SD1.5/PyTorch setting.

Academic reference: Juan M. Cardenas, Ben Adcock, and Nick Dexter,
"CS4ML: A general framework for active learning with arbitrary data based on
Christoffel functions," NeurIPS 2023.
[OpenReview](https://openreview.net/forum?id=aINqoP32cb)

MIT License

Copyright (c) 2024 Juan Manuel Cardenas Cardenas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
