# Automatic Comparison of Concepts with Relational Descriptions (ACCoRD)

A single scientific concept can be described in many different ways, benefiting differences in individuals’ background knowledge. However, current automatic concept description generation systems only produce a single “best” description in the context of a single document.

To solve this problem, ACCoRD makes use of the variety of ways a concept is talked about across the scientific literature to produce multiple descriptions of a given target concept, in terms of distinct reference concepts. Our system automatically extracts the relevant text from the scientific literature and produces a succinct, self-contained summary of the concept description.

This repository contains the code, demo link, and dataset for the ACCoRD system. See our arXiv preprint for system implementation and details.

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
