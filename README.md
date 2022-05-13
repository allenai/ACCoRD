# Automatic Comparison of Concepts with Relational Descriptions (ACCoRD)

Systems that can automatically define unfamiliar terms promise to improve the accessibility of scientific texts, especially for readers who may lack prerequisite background knowledge. However, current systems assume a single “best” description per concept, which fails to account for the many potentially useful ways a concept can be described.

To address this problem, we introduce ACCoRD, an end-to-end system tackling the novel task of generating sets of descriptions of scientific concepts. Our system takes advantage of the myriad ways a concept is mentioned across the scientific literature to produce distinct, diverse descriptions of target scien- tific concepts in terms of different reference concepts. To support research on the task, we release an expert-annotated resource, the ACCoRD corpus.

This repository contains the code, demo link, and dataset for the ACCoRD system. See our arXiv preprint for system implementation and dataset details.

ACCoRD is maintained by [Sonia Murthy](https://github.com/skmur) at the [Allen Institute for AI](https://allenai.org/).

## System demonstration

The output of the ACCoRD system for 150 popular Natural Language Processing (NLP) concepts can be found at [accord.allenai.org](https://accord.allenai.org/)

## Corpus

The ACCoRD corpus is a high-quality, expert-annotated resource for the task of producing multiple descriptions of a single scientific concept in terms of distinct reference concepts.

* ACCoRD includes 1275 labeled contexts and 1787 hand-authored concept descriptions from 698 computer science papers from [S2ORC](https://github.com/allenai/s2orc).
    * Extractions were labeled as positive if they described a target ForeCite concept interms of any other concept.
    * Each positive extraction was allowed to have multiple concept descriptions if the target concept was described in terms of multiple other concepts in the source text, or if the extraction contained multiple target concepts.

* We include annotations for both 1-sentence and 2-sentence source text settings to allow for experiments on the effects of richer context on our task.

## License

ACCoRD is released under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/legalcode). By using ACCoRD, you are agreeing to its usage terms.
