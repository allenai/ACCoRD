# Automatic Comparison of Concepts with Relational Descriptions (ACCoRD)

ACCoRD is a high-quality (Cohen’s κ= 0.658), expert-annotated resource for the task of producing multiple descriptions of a single scientific concept in terms of distinct reference concepts.

* ACCoRD includes 3336 labeled extractions and 1785 hand-authored concept descriptions from 1387 computer science papers from [S2ORC](https://github.com/allenai/s2orc).
    * Extractions were labeled as positive if they described a target ForeCite concept interms of any other concept.
    * Each positive extraction was allowed to have multiple concept descriptions if the target concept was described in terms of multiple other concepts in the source text, or if the extraction contained multiple target concepts.

* We include annotations for both 1-sentence and 2-sentence source text settings to allow for experiments on the effects of richer context on our task.

ACCoRD is maintained by [Sonia Murthy](https://github.com/skmur) at the [Allen Institute for AI](https://allenai.org/).

## License

ACCoRD is released under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/legalcode). By using ACCoRD, you are agreeing to its usage terms.
