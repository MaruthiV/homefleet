# Research README references

Updated 2026-11-17. The initial references were inspected November10; the November17 presentation pass also follows the two research repositories supplied by the author.

| Primary source | Pattern adopted |
| --- | --- |
| [LLaDA README](https://github.com/ML-GSAI/LLaDA) | Paper and release badges, concise introduction, figures near the top, direct usage commands, and citation. |
| [PrefixKV README](https://github.com/THU-MIG/PrefixKV) | Paper title and authors, an overview diagram, a collapsible abstract, result figures, environment setup, and evaluation commands. |
| [SustainGym README](https://github.com/chrisyeh96/sustaingym#readme) | A concise description, paper and documentation links, repository map, license, and citation. |
| [CityLearn README](https://github.com/citylearn-project/CityLearn#readme) | A short runnable simulation with public inputs and interpretable outputs. |
| [safe-control-gym README](https://github.com/learnsyslab/safe-control-gym#readme) | Publications mapped to reproducible code versions, installation details, and explicit example commands. |
| [CleanRL README](https://github.com/vwxyzjn/cleanrl#readme) | Concrete seeded training commands, dependencies, logging, and benchmark links. |

HomeFleet uses two entry paths: an included eight-home example that requires no download, and a versioned paper reproduction path with exact input bundles, checkpoints, configurations, and output tables. The README distinguishes simulation inputs from observations, implemented controllers from prospective extensions, and a source rebuild from reproduction using frozen inputs.

The public manuscript uses the [NeurIPS preprint option](https://neurips.cc/Conferences/2026/MainTrackHandbook). Its source archive follows [arXiv's TeX submission instructions](https://info.arxiv.org/help/submit_tex.html), with required style, figure, and bibliography files and a clean independent build.

The revised HomeFleet landing page leads with the paper title, authors/affiliations and paper/artifact links, an original SVG overview of the actual replay pipeline, and the existing verified result figures. Full assumptions and experiment details remain in the reproduction guide and data card. Generated headline and compute blocks remain byte-identical; the design introduces no new scientific results or claims of conference acceptance/arXiv assignment.
