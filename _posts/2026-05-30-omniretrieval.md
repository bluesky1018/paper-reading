---
layout: post
title: "OmniRetrieval：跨异构知识源的统一检索"
date: 2026-05-30
categories: [论文解读, RAG检索]
tags: ["检索增强生成", "RAG", "知识图谱", "向量检索", "统一模型"]
---

> 📄 **论文**：OmniRetrieval: Unified Retrieval across Heterogeneous Knowledge Sources
> 🔗 **arXiv**：[2605.29250](https://arxiv.org/abs/2605.29250)
> 🏢 **机构**：

## 一句话总结

Real-world information needs require access to structurally diverse knowledge sources, from unstructured text and relational tables to knowledge graphs and property graphs. Existing retrievers, howeve...

## 背景与问题

The knowledge that answers a real-world question rarely lives in a single place, or in a single shape. A clinical question may be answered by a passage in a biomedical article ( BEIR ) ; an enterprise question may require a join across normalized relational tables ( Spider ; Bird ) ; a factoid question about people, places, or events may resolve to a few triples in an encyclopedic knowledge graph ( Freebase ; Wikidata ) ; and a question about a supply chain or an academic collaboration network may turn on a multi-hop traversal of a labeled property graph ( Text2Cypher ) . In each case, the right answer is, in principle, retrievable, but only if one already knows which corpus to consult, which query language to write, and which execution engine to dispatch it to. The retrieval problem, then, is not merely to find relevant content within a source, but to navigate the structural heterogeneity that runs across sources.

Existing retrieval approaches, however, are typically designed for one source at a time. Specifically, document retrievers operate over an unstructured corpus and rank passages by similarity to a free-form query ( BM25 ; DPR ) ; text-to-SQL systems target a single relational database and emit a single SQL dialect ( Spider ; Bird ) ; SPARQL or Cypher generators are likewise tied to a single graph backend and query language, with SPARQL for RDF stores and Cypher for labeled property graphs ( Text2SPARQL ; Text2Cypher ) . As a consequence, even when a recent Large La


![Figure 1 : Different knowledge sources offer distinct structural affordances and query languages (le](https://arxiv.org/html/2605.29250/2605.29250v1/x1.png)
*图：Figure 1 : Different knowledge sources offer distinct structural affordances and query languages (le*


![Figure 2 : Effect of the candidate list size k k in source selection on Source Selection and Retriev](https://arxiv.org/html/2605.29250/2605.29250v1/x2.png)
*图：Figure 2 : Effect of the candidate list size k k in source selection on Source Selection and Retriev*


Classical retrieval has long been organized around a single corpus and representation, from lexical retrievers that rank passages by term overlap ( BM25 ; tfidf ) to dense retrievers that project queries and documents into an embedding space ( DPR ; ANCE ) , with multi-modal extensions adding specific encoders for images or video ( CLIP ; VideoRAG ) . To lift this single-source restriction, one line of work collapses heterogeneous sources, such as text passages, knowledge graph facts, and tabular records, into a shared representation so that a single retriever can rank items across them ( UniK ; UDT ; DiFaR ) . In the meantime, other efforts cover either structured or unstructured sources but not both: query-type routers are confined to unstructured corpora and rely on embedding similarity

## 核心方法

Let q q be a question from a user and ℬ = { b 1 , … , b N } \mathcal{B}=\{b_{1},\ldots,b_{N}\} be a pool of independently maintained knowledge sources. Each of these sources b ∈ ℬ b\in\mathcal{B} has its own native query language (such as SQL for a relational database, SPARQL for an RDF graph, Cypher for a labeled property graph, or free-form text for an unstructured corpus), its own execution engine Exec ​ ( b , q ^ ) \texttt{Exec}(b,\hat{q}) that accepts a native query q ^ \hat{q} (written in that language) and returns a set of results, and an exposed structural context c b c_{b} (such as a relational schema, an ontology, or a corpus descriptor) that any external caller can read in order to formulate an executable query against b b . However, knowledge sources may differ arbitrarily in what they store and how they store it, where one may hold unstructured text, another normalized tables, and a third a labeled graph, and they return their results in correspondingly different forms.

The retrieval task is then to find and provide, for the question q q , a set of evidence drawn from one or more sources in ℬ \mathcal{B} that is relevant to q q . Notably, a retrieval framework addressing this task should operationalize the selection of a subset 𝒮 ⊆ ℬ \mathcal{S}\subseteq\mathcal{B} of sources to engage, the formulation of an executable query q ^ b \hat{q}_{b} in the native language of each b ∈ 𝒮 b\in\mathcal{S} , and the consolidation of the executor outputs { Exec ​ ( b , q ^ b ) } b ∈ 𝒮 \{\texttt{Exec}(b,\hat{q}_{b})\}_{b\in\mathcal{S}} into a single evidence set relevant to q q . This formulation has clear strengths. In particular, since each source is engaged through its own native language, the structural operators it exposes (such as joins, traversals, property paths) are preserved rather than approximated by similarity in a shared space. Also, keeping each source on its own terms makes adding a new source a matter of registration rather than infrastructure rebui


![Figure 3 : Evidence-selection accuracy on multi-candidate questions with the gold in the top- k k .](https://arxiv.org/html/2605.29250/2605.29250v1/x3.png)
*图：Figure 3 : Evidence-selection accuracy on multi-candidate questions with the gold in the top- k k .*


![Figure 4 : Effect of backbone scale (Qwen-3.5, 2B to 27B). Oracle (Gold Source) uses the gold source](https://arxiv.org/html/2605.29250/2605.29250v1/x4.png)
*图：Figure 4 : Effect of backbone scale (Qwen-3.5, 2B to 27B). Oracle (Gold Source) uses the gold source*


![Figure 5 : Candidate diversity: distinct retrieval paradigms and knowledge sources per sample.](https://arxiv.org/html/2605.29250/2605.29250v1/x5.png)
*图：Figure 5 : Candidate diversity: distinct retrieval paradigms and knowledge sources per sample.*


![Figure 6 : Source-selection behavior on 1 and 2+ candidate regimes. Solid segments mark success (gol](https://arxiv.org/html/2605.29250/2605.29250v1/x6.png)
*图：Figure 6 : Source-selection behavior on 1 and 2+ candidate regimes. Solid segments mark success (gol*


## 实验结果

We evaluate OmniRetrieval on a benchmark compiled from 13 datasets that, in combination, span all four native backends, and that together provide a pool of 309 distinct knowledge bases.

For document retrieval over unstructured corpora, whose task is to identify documents that are most relevant to a natural-language query, we use seven datasets of various domains from the BEIR benchmark ( BEIR ) : NFCorpus (medical) ( NFCorpus ) , SciFact (scientific claim verification) ( SciFact ) , FiQA ( FiQA ) (financial question answering), MS MARCO (web passages) ( MSMARCO ) , FEVER (Wikipedia fact verification) ( FEVER ) , Natural Questions (short-answer question answering) ( NQ ) , and HotpotQA ( HotpotQA ) (multi-hop question answering). Each document collection itself serves as a knowledge base.


![Figure 7 : LLM-Judge accuracy on GPT-5.4 (rows: method paradigm; columns: question paradigm).](https://arxiv.org/html/2605.29250/2605.29250v1/x7.png)
*图：Figure 7 : LLM-Judge accuracy on GPT-5.4 (rows: method paradigm; columns: question paradigm).*


![Figure 8 : Predicted retrieval paradigm distribution under per-paradigm balanced weighting. Left: to](https://arxiv.org/html/2605.29250/2605.29250v1/x8.png)
*图：Figure 8 : Predicted retrieval paradigm distribution under per-paradigm balanced weighting. Left: to*


## 全文图示

## 总结

In this work, we presented OmniRetrieval, a framework for retrieval over structurally heterogeneous knowledge sources that, given a natural-language question, engages each relevant source through its own native query language and consolidates the executor outputs via a cross-source evidence selection step, rather than collapsing the sources into a shared representation. Evaluation on a benchmark spanning 13 datasets and 309 knowledge bases over unstructured corpora, relational databases, RDF graphs, and labeled property graphs shows that OmniRetrieval consistently outperforms relevant baselines. Our analyses further indicate that broad exploration at the source-selection step, with the final commitment deferred to a selector that rests on retrieved evidence, is what lets OmniRetrieval scale gracefully. These findings position OmniRetrieval as a step toward a general-purpose universal layer, one that preserves the structural affordances that make each source valuable while exposing a si

