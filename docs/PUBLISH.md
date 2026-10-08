# Publish as matinfirooz/AttentionForge-NumPy

Create an empty GitHub repository named **AttentionForge-NumPy** under
`matinfirooz`. Leave GitHub's initial README, license, and .gitignore options
unchecked because this package already includes those files.

Extract the package and run these commands inside its `AttentionForge-NumPy`
directory. They require Git and authentication to your GitHub account.

```bash
git init -b main
git add .
git commit -m "Build NumPy attention from forward pass to learned retrieval"
git remote add origin https://github.com/matinfirooz/AttentionForge-NumPy.git
git push -u origin main
```

Use this repository description:

> NumPy-only attention from scratch: manual backpropagation, multi-head attention, RoPE, KV caching, tiled online softmax, and an offline visual explorer.

Suggested GitHub topics:

`numpy` `attention-mechanism` `self-attention` `multi-head-attention`
`from-scratch` `machine-learning` `transformers` `kv-cache` `rope` `educational`

The README uses relative image links and GitHub math delimiters. All diagrams,
result files, and the trained teaching checkpoint are included. The CI
workflow will start on the first push; its matrix has not run until that push.

To showcase the explorer, download and open
`docs/attention_explorer.html`. GitHub's file viewer displays HTML source rather
than executing it. The file itself is self-contained and works offline.

If you use GitHub's web uploader, upload the **contents** of the extracted
project directory so `README.md` is at the repository root. Upload the hidden
`.github` directory and `.gitignore` as well; pushing with Git handles them.
