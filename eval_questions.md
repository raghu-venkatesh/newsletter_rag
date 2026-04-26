# RAG Evaluation Questions

50 questions for evaluating retrieval quality, answer faithfulness, citation accuracy, and failure modes.

---

## Factual / Single-Source Retrieval
*Tests: does the right chunk get fetched?*

1. What are the four key eras in AI development covered in the History of AI newsletter?
2. What were the main limitations of expert systems in the 1970s and 1980s?
3. What skills and tools define an AI Engineer according to the nocode.ai newsletter?
4. How does a Mixture of Experts (MoE) model work, and what makes it compute-efficient?
5. What are the four stages of human-machine collaboration Tyler Cowen describes in *Average Is Over*?
6. What does Scott Young conclude about audiobooks vs. paper books for retention?
7. What was the subscriber count of nocode.ai when the History of AI issue was sent?
8. What was the Amazon S3 security update described in newsletter #124?
9. What Google product announcement appeared alongside the Amazon S3 story in issue #124?
10. What is the Voynich manuscript, according to the newsletter about it?

---

## Specific Details / Dates
*Tests: precision, no hallucination of specifics*

11. What exact date was "The Rise of the AI Engineer" newsletter published?
12. What was the subscriber count of nocode.ai when the MoE newsletter was sent?
13. What is SageMaker Fridays and how frequently did the sessions run?
14. When was Fei-Fei Li's talk on ImageNet and Visual Intelligence scheduled?
15. What Facebook policy updates were described in newsletter #138, and what problem were they meant to solve?

---

## Multi-Document Synthesis
*Tests: can the system combine across chunks/sources?*

16. What do the newsletters collectively say about the future of middle-class white-collar jobs?
17. How do the newsletters collectively describe the career transition from software engineer to AI engineer?
18. What learning platforms or courses are recommended across the corpus?
19. What security-related topics appear across multiple newsletters?
20. What do the newsletters say about the relationship between machine learning and deep learning?

---

## Comparison / Reasoning
*Tests: synthesis, not just lookup*

21. How do expert systems differ from deep learning models, based on what the newsletters describe?
22. According to the newsletters, what separates high earners from low earners in an automated economy?
23. What are the trade-offs of sparse MoE models vs. dense models as described in the newsletter?
24. How does Tyler Cowen's take on automation compare to Scott Young's advice on personal learning?
25. What does Scott Young say about the effectiveness differences between Kindle and paper reading?

---

## Attribution / Source Citation
*Tests: citation brackets are accurate and traceable*

26. Which newsletter and author discussed *Average Is Over* as a book club pick?
27. Which newsletter covers Natural Language Understanding in Alexa, and who sent it?
28. Which newsletter discusses CAPTCHAs — what's the framing, and who is the sender?
29. What does Benedict's Newsletter typically cover, based on the issues in the corpus?
30. Which newsletters in the corpus were sent from AWS, and what was the recurring format?

---

## Out-of-Scope Questions
*Tests: graceful refusal, no hallucination*

31. What is the current price of Bitcoin?
32. What does the corpus say about quantum computing?
33. Who won the 2024 US presidential election?
34. What are the latest updates to GPT-4o?
35. What does the newsletter say about Rust as a programming language?

---

## Ambiguous Queries
*Tests: clarification behavior and retrieval robustness*

36. What do the newsletters say about "transformers"? *(ambiguous: AI transformers vs. electrical component)*
37. What is covered about "Python" in the corpus? *(could match programming or non-programming content)*
38. What does the corpus say about "fine-tuning"?
39. Tell me about "experts" mentioned in the newsletters.
40. What do the newsletters say about "chains"? *(LangChain vs. supply chains vs. other uses)*

---

## Hallucination Traps / Specificity Stress Tests
*Tests: model stays grounded, doesn't invent*

41. What specific salary figures for AI Engineers are mentioned in the newsletters?
42. What exact date did Google announce the TensorFlow release referenced in newsletter #124?
43. What exact subscriber count did nocode.ai have when the Rise of the AI Engineer issue was sent?
44. What specific studies on mind-wandering does Scott Young cite in his reading formats newsletter?
45. What exact AWS services were demonstrated during SageMaker Fridays sessions?

---

## Broad / Aggregation Queries
*Tests: handling large scope, top-k=4 chunk limit pressure*

46. Summarize everything the corpus says about Foundation Models.
47. What books or book clubs are mentioned across all newsletters in the corpus?
48. What does the corpus say about web frameworks — React, Angular, GraphQL, and Node.js?
49. List all NLP-related topics mentioned across the newsletters.
50. What personal finance topics appear in the corpus, and which newsletters cover them?

---

## What Each Category Reveals

| Category | What it exposes |
|---|---|
| Factual / Single-source | Embedding quality, chunk boundary decisions |
| Specific details / Dates | Whether metadata (date, author) is preserved in chunks |
| Multi-doc synthesis | top_k=4 ceiling — may need to raise it |
| Comparison / Reasoning | Generator quality with llama3.2:3b |
| Attribution / Citation | Whether `[1]`, `[2]` brackets map to correct sources |
| Out-of-scope | Prompt grounding — does it refuse or hallucinate? |
| Ambiguous queries | Embedding space behavior under polysemy |
| Hallucination traps | Whether the model invents plausible-sounding specifics |
| Broad / Aggregation | Hard ceiling of top_k=4 likely causes incomplete answers |
