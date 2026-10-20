# Practical AI in Industry

[![License: GPL-3.0](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Slide Deck](https://img.shields.io/badge/Slides-Marp-0288d1.svg)](presentation.html)
[![Tech Stack](https://img.shields.io/badge/Focus-MLOps%20%7C%20LLMOps%20%7C%20FastAPI-orange.svg)](#modern-industry-tech-stack)

> **"In coursework, the model is the finish line. In industry, a trained model is only the starting line."**

A practical slide deck and reference guide bridging the gap between academic/coursework machine learning and production-grade AI systems. This repository covers real-world use cases, system architectures, deployment patterns, modern MLOps/LLMOps stacks, and actionable career advice for aspiring Machine Learning Engineers.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Topics & Agenda](#-key-topics--agenda)
- [Modern Industry Tech Stack](#-modern-industry-tech-stack)
- [Career Preparation & Advice](#-career-preparation--advice)
- [How to View and Build the Slides](#-how-to-view-and-build-the-slides)
- [Speaker & Attribution](#-speaker--attribution)
- [License](#-license)

---

## 🎯 Overview

Most AI in production isn't a flashy demo or a standalone Jupyter Notebook; it is a quiet, reliable software service making millions of automated decisions daily to **reduce operational costs**, **grow revenue**, and **enhance user experience**.

This repository provides:
- **Presentation Source**: Markdown slides styled and formatted for [Marp](https://marp.app/) (`presentation.md`).
- **Interactive Web Presentation**: Standalone, responsive HTML presentation with built-in presenter notes and timer (`presentation.html`).

---

## 📌 Key Topics & Agenda

1. **Data as the Strategic Asset**: Moving from raw data to refined business intelligence and automated decision engines.
2. **Business Objectives of AI**:
   - *Cost Reduction*: Automating repetitive operations (customer support, moderation).
   - *Revenue Generation*: Personalization, search ranking, dynamic pricing, and conversion optimization.
   - *Customer Experience*: Low-latency, contextual user experiences.
3. **Real-World Case Studies (E-Commerce & OTA)**:
   - Search autocomplete & ranking engines
   - Cross-sell recommender systems
   - Automated review categorization & sentiment classification
   - Scalable image curation & quality checks
   - 24/7 intelligent customer care chatbots
4. **The Reality of ML Engineering**:
   - The model accounts for **~10–20%** of the work.
   - **~80%** is robust software engineering: pipelines, testing, containerization, serving, and continuous monitoring.
5. **Deployment Patterns**:
   - *Synchronous (Real-time)*: Low latency, user-facing (FastAPI, Redis, gRPC).
   - *Asynchronous (Batch)*: High throughput offline scoring (Airflow, Spark, dbt).
   - *Scheduled / Event-driven*: Recurring retraining pipelines and drift checks.

---

## 🛠 Modern Industry Tech Stack

| Category | Tools & Technologies |
| :--- | :--- |
| **Infrastructure & Cloud** | AWS, GCP, Azure, Kubernetes, Consul, Vault, ArgoCD |
| **Databases & Vector Stores** | PostgreSQL, MongoDB, Redis, Milvus, FalkorDB |
| **Data Pipelines** | Apache Airflow, Kubeflow, Apache Beam, Google Dataflow, dbt |
| **MLOps & LLMOps** | MLflow, AI Gateways, Prompt Registries, Model Versioning |
| **Serving & APIs** | FastAPI, Pydantic, Docker, uv |
| **Machine Learning & Frameworks** | PyTorch, scikit-learn, TensorFlow, Polars, Pandas |
| **Agentic Frameworks** | Google ADK, Microsoft Agent Framework, Pydantic AI, CrewAI |
| **Observability** | Grafana, Kibana, Prometheus |

---

## 🚀 Career Preparation & Advice

Practical guidelines for students and engineers entering the field:

- **Master the Fundamentals First**: Strong Software Engineering principles (Data Structures, Algorithms, SQL, System Design) beat chasing every new framework.
- **Fluency in Essential Tooling**: Git, Docker, and the Linux CLI are non-negotiable.
- **Ship 2–3 End-to-End Projects**: Go beyond notebooks. Take raw data $\rightarrow$ train a model $\rightarrow$ package into Docker $\rightarrow$ expose as a FastAPI endpoint $\rightarrow$ deploy to cloud.
- **Pursue Internships**: Hands-on exposure to production constraints, code reviews, and cross-functional teams is the strongest differentiator on a resume.
- **Communication & Growth Mindset**: Learn to explain complex trade-offs to non-technical stakeholders and adapt as tools evolve.

---

## 🖥 How to View and Build the Slides

### 1. View Directly in Browser
Simply open the included `presentation.html` file in any modern web browser:
```bash
# On Linux
xdg-open presentation.html

# On macOS
open presentation.html
```
- Press **`F`** for Fullscreen.
- Press **`P`** for Presenter Mode (includes speaker notes and timer).
- Press **`O`** for Slide Overview.

### 2. Edit or Export using Marp CLI
You can customize `presentation.md` and rebuild the output formats using [Marp CLI](https://github.com/marp-team/marp-cli):

```bash
# Start local live preview server
npx @marp-team/marp-cli -s .

# Export to HTML
npx @marp-team/marp-cli presentation.md -o presentation.html

# Export to PDF (requires Chromium)
npx @marp-team/marp-cli presentation.md --pdf -o presentation.pdf

# Export to PowerPoint (PPTX)
npx @marp-team/marp-cli presentation.md --pptx -o presentation.pptx
```

---

## 👤 Speaker & Attribution

- **Speaker**: **Setia Budi**
  - Machine Learning Engineer at **tiket.com**
- **Original Topic**: *A Brief Story of AI Implementation in Industry*

---

## 📄 License

This repository is distributed under the [GNU General Public License v3.0](LICENSE).
