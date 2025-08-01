## HACTREE: HS Code Auto Classification and Trade Intelligence System for Export-Oriented Enterprises

### 1. Project Overview

**HACTREE** is an AI-powered export strategy analyst platform designed to assist enterprises preparing for international trade. It offers automated HS code classification, integrated access to country-specific tariff and TBT (Technical Barriers to Trade) data, promising market analysis, and auto-generated export strategy reports.

---

### 2. Project Description

#### Objective

This project aims to develop an AI-driven platform that automates the end-to-end process of export strategy formulation by providing the following capabilities:

* Automatic classification of Harmonized System (HS) codes based on product descriptions
* Retrieval and presentation of relevant tariff and non-tariff (TBT) information by country
* Recommendation of promising export destinations
* Automated generation of customized export strategy reports

#### Background and Motivation

* Exporters often face difficulties in interpreting and consolidating complex regulatory and tariff data across multiple jurisdictions
* Although public trade datasets are available, the burden of interpretation and strategy development remains on the user
* There is a growing demand for AI services that facilitate strategic decision-making without requiring expert knowledge
* This project further develops the award-winning idea "TREE" (2023 competition) into a practical and scalable solution

#### Technologies and Tools

* **HS Code Classification**: BiLSTM-based text classifier trained on official customs datasets
* **Natural Language QA**: Retrieval-Augmented Generation (RAG) system using Polyglot-ko LLM and FAISS-based vector search
* **Document Processing**: pdfplumber, Tesseract OCR, LangChain TextSplitter
* **Text Embedding**: SentenceTransformer (MiniLM) and local FAISS vector database
* **Web Interface**: Interactive UI developed with Gradio
* **Development Environment**: Python 3.10, Visual Studio Code

#### Key Features

* HS code prediction based on input product names or descriptions
* Automatic linkage to tariff rates and TBT requirements based on predicted HS codes
* Real-time, vector-based QA system for trade strategy, market information, and regulatory insights
* Market attractiveness analysis and automated generation of export strategy reports by country
* Future enhancements include customizable PDF/HTML report storage and distribution

#### Business Strategy and Expected Impact

* **Scalable Development Roadmap**

  * *Phase 1*: Open-source prototype for export strategy analysis
  * *Phase 2*: Expansion into a SaaS platform (user accounts, team collaboration, API access)

* **Revenue Model**

  * Paid report generation services for B2B and B2G clients
  * Subscription-based API access to trade intelligence data

* **Anticipated Societal Impact**

  * Empowerment of SMEs to independently formulate digital export strategies
  * Improved usability of public trade data
  * Reduction in export failures through automated HS code validation and TBT risk alerts
