---
marp: true
theme: uncover
class: invert
paginate: true
backgroundColor: #0d1117
color: #e6edf3
style: |
  section {
    background: linear-gradient(160deg, #0d1117 0%, #161b22 100%);
    color: #e6edf3;
    font-size: 30px;
    text-align: left;
    justify-content: flex-start;
    padding: 60px 70px;
  }
  h1 {
    color: #58a6ff;
    font-size: 54px;
  }
  h2 {
    color: #58a6ff;
    border-bottom: 2px solid #30363d;
    padding-bottom: 12px;
  }
  h3 { color: #79c0ff; }
  strong { color: #ffa657; }
  em { color: #7ee787; }
  a { color: #58a6ff; }
  blockquote {
    border-left: 4px solid #58a6ff;
    background: #161b22;
    padding: 10px 20px;
    color: #adbac7;
    font-style: italic;
  }
  ul, ol { line-height: 1.5; }
  code {
    background: #161b22;
    color: #ffa657;
    padding: 2px 6px;
    border-radius: 4px;
  }
  section.lead { text-align: center; justify-content: center; }
  section.lead h1 { font-size: 60px; }
  footer { color: #6e7681; }
  section::after {
    color: #6e7681;
    font-size: 18px;
  }
---

<!--
Format note: Each "---" separates a slide. Text inside HTML comments (like this one)
is a presenter note — visible in Marp presenter view but NOT on the slide.
To export: run `marp presentation.md --pdf` (or --pptx / --html), or use the Marp for VS Code extension.
-->

<!-- _class: lead invert -->

# A Brief Story of AI Implementation in Industry

**Setia Budi**
Machine Learning Engineer at tiket.com

<!--
Keep it warm and personal for the first 30 seconds — you want the room relaxed.
Set expectations: "By the end, you'll know what an ML engineer really does day-to-day,
and 3–4 concrete things you can start doing this semester."
-->

---

## Introduction

- Hi everyone, my name is **Setia Budi**.
- I'm a **Machine Learning Engineer at tiket.com** — one of the most loved Online Travel Agents in Indonesia.
- Today's goal: a real, practical look at how **AI/ML actually works in industry** — and how to prepare for it before you graduate.
- Please **interrupt and ask questions at any point.** This is a conversation, not a lecture.

<!--
Ask a quick show of hands: "Who here has trained a model in a course project?"
— this tells you how technical to go.
-->

---

## Agenda

1. Why data (and AI) matters in business
2. What AI in industry actually looks like
3. The role of an ML/AI Engineer
4. How AI services are deployed
5. The common industry tech stack
6. Fundamental skills that matter
7. **How to prepare before graduation**
8. Q&A

<!--
Signpost the flow so students know the "career prep" payoff is coming
— it keeps them engaged through the technical middle.
-->

---

## Data is the New Oil

- Ever heard the slogan **"Data is the new oil"**?
- What does it actually mean?
  - Like oil, **raw data is not valuable by itself** — it must be *refined*.
  - Refining = collecting, cleaning, storing, turning data into **decisions and products**.
  - The companies that refine data best **win their markets**.

<!--
Ask the room what they think it means before revealing.
Example: "Every search you type, every hotel you click, every booking you cancel
— that's crude oil. Our job is to refine it into recommendations, pricing, and fraud detection."
Modern nuance: today it's less about HAVING data and more about USING it responsibly and fast.
-->

---

## AI in the Industry

- Is AI in industry **for real**, or just hype?
- Short answer: **very real** — but often invisible.
- Most production AI isn't a flashy robot; it's a quiet service making **millions of small decisions per day**.
- Where is AI already working behind the scenes in an app you used *today*?

<!--
Let students shout out examples (feeds, maps, spam filters, recommendations).
Key reframe: "The best AI in production is the AI you never notice."
Distinguish hype (demos) from value (reliable systems that move a business metric).
-->

---

## Objectives of AI in Industry

At the end of the day, business AI serves **two goals**:

- **Reduce operational cost** — automate repetitive human work (support, moderation, data entry).
- **Improve revenue** — better search, recommendations, pricing, and conversion.

A third, underrated one: **improve customer experience** (which feeds back into both).

<!--
Most important mindset slide for students: a model that doesn't move a business metric doesn't ship.
Accuracy is not the goal — IMPACT is. A 99% accurate model nobody uses is worth less
than a 90% model that saves the company money. Tie every later example back to "cost" or "revenue."
-->

---

## A Glimpse of AI in Industry

Real examples from an online travel / e-commerce context:

- **Search keyword autocomplete** — predict what the user wants to type
- **Product / hotel search ranking** — order results to maximize relevance & conversion
- **Cross-sell recommender** — "customers who booked this also booked…"
- **Review categorization** — auto-tag and summarize reviews
- **Image curation** — detect low-quality or wrong images at scale
- **Smart chatbot** — handle common support 24/7
- **Email / WhatsApp automation** — personalized/standardized, well-timed messaging

<!--
For each, quickly name the business goal: ranking → revenue, chatbot → cost, image curation → experience.
Pick ONE you've personally worked on and tell a 30-second story — real experience is what students remember.
-->

---

## The Role of an ML / AI Engineer

- **Develop production-ready AI services** — not notebooks, but reliable software.
- **Handle deployment** of those services into live systems.
- **Monitor** them after launch (performance, drift, cost, failures).

> The model is maybe **10–20%** of the job. The other 80% is **engineering**: data, pipelines, serving, testing, and monitoring.

<!--
Biggest surprise for students: industry ML ≠ Kaggle.
"In coursework, the model is the finish line. In industry, a trained model is the STARTING line."
An ML engineer owns the whole lifecycle, not just model.fit().
-->

---

## How AI Services Are Deployed

- **Synchronous (user-facing)** — real time, e.g. search ranking, autocomplete. Low latency is critical.
- **Asynchronous (batch)** — process large volumes offline, e.g. scoring millions of items overnight.
- **Scheduled** — runs daily / weekly / monthly, e.g. recomputing recommendations or retraining.

<!--
Mental model: "Do we need the answer NOW (sync), SOON in bulk (batch), or ON A CLOCK (scheduled)?"
Trade-offs: sync = strict latency & scaling; batch = throughput & cost efficiency.
-->

---

## Common AI Tech Stacks

The moving parts behind a production AI system:

- **Infrastructure** — where things run
- **Data Pipeline** — how data moves
- **MLOps** — how models are managed
- **LLMOps** — how LLM apps are managed
- **Development Tools** — how we build day-to-day

<!--
Reassure them: "You don't need to master all of these before graduating — but you should
RECOGNIZE them so you're not lost on day one." Next slides zoom into each.
-->

---

## Infrastructure

- **Cloud:** AWS, GCP, Azure, Alibaba Cloud
- **Databases:** PostgreSQL, MongoDB, Redis, Milvus, FalkorDB *(vector & graph stores for AI/RAG)*
- **Container orchestration:** Kubernetes
- **Deployment / delivery:** Jenkins, ArgoCD, Consul, Vault
- **Monitoring:** Grafana, Kibana

<!--
Don't read the whole list — call out the CATEGORIES: compute, storage, orchestration, delivery, observability.
Highlight that vector databases (Milvus) are newly essential because of LLMs and semantic search.
Advice: "Pick ONE cloud and get comfortable — the concepts transfer."
-->

---

## Data Pipeline

- **Batch pipelines:** Kubeflow, Apache Airflow, dbt
- **Streaming pipelines:** Apache Beam, Google Dataflow

<!--
Unglamorous truth: most ML work is data plumbing.
Batch = "process a big chunk on a schedule." Streaming = "process events as they arrive."
If a student learns just one thing here: SQL + Airflow — universally useful.
-->

---

## MLOps

- **Experiment tracking** — record every run, parameter, and metric for reproducibility.
- **Model registry** — version, stage, and promote models (dev → staging → prod).
- **Platform example:** MLflow

<!--
Define MLOps simply: "DevOps, but for machine learning."
Without tracking, you can't answer "which model is in production and how was it trained?"
— a real, scary question.
-->

---

## LLMOps

- **AI Gateway** — a single controlled entry point to LLM providers (routing, rate limits, cost, security).
- **Prompt registry** — version and manage prompts like code.
- **Platform example:** MLflow (with growing LLM support)

<!--
This category barely existed a few years ago — a great example of how fast this field moves.
"If you build with GPT/Claude APIs, you're already doing early LLMOps."
-->

---

## Development Tools

- **Editor / IDE:** Jupyter, VS Code, Vim
- **Version control:** Git *(non-negotiable)*
- **Containerization:** Docker
- **AI coding assistants:** Cursor, Claude Code, Copilot

<!--
Two must-haves TODAY: Git and Docker. Learn them now, not on the job.
On AI assistants: "Use them to go faster, but understand what they produce — you own the code, not the tool."
-->

---

## Python Toolbox

- **Env / package manager:** uv *(fast, modern replacement for pip + venv)*
- **API framework:** FastAPI *(how models become services)*
- **Data validation:** Pydantic
- **Testing:** PyTest
- **DataFrames:** Pandas, Polars
- **AI / ML:** scikit-learn, PyTorch, TensorFlow, etc
- **AI agents:** Google ADK, Microsoft Agent Framework, Pydantic AI, CrewAI

<!--
Python is the lingua franca of ML — get genuinely comfortable here.
Priorities for students: Pandas + scikit-learn first, then FastAPI + Pydantic to ship a model as an API.
Polars and uv show the ecosystem constantly evolving toward speed.
The AI-agent tools are the newest frontier — worth exploring, but master the fundamentals first.
-->

---

## Fundamental Skills for an ML / AI Engineer

- **Software Engineering** — prolem solving, clean code, testing, Git, APIs, Docker.
- **Learning New Things (Growth Mindset)** — the stack changes; the ability to *keep learning* is the real skill.
- **Communication** — explain trade-offs to non-technical people; a model nobody understands won't be trusted or used.

<!--
Rank them honestly: "If I had to hire one trait, it's the growth mindset — everything else can be learned."
ML engineers are software engineers FIRST. Strong SWE fundamentals beat fancy model knowledge.
-->

---

## How to Prepare Before You Graduate

Do-it-this-semester advice:

- **Master the fundamentals** — data structures, algorithms, *especially* SQL.
- **Get fluent in Git, Docker, and the Linux command line.**
- **Build 2–3 end-to-end projects** — data → model → *deployed API*. Deployment impresses employers.
- **Put it on GitHub** with a clean README. Your public work is your resume.
- **Do an internship** — nothing beats *real industry experience* before you graduate.
- **Learn to read docs and debug independently.**
- **Write and speak clearly** — practice explaining your projects in 2 minutes.

<!--
This is the slide they'll photograph — slow down and let it land.
Reframe grades: "Employers care more about what you can BUILD than your GPA."
One deployed project > ten half-finished tutorials.
On internships: emphasize this strongly — a real internship exposes you to production systems,
teamwork, code review, and how decisions are really made. It's the single biggest added value
a fresh graduate can walk in with, and often the fastest path to a full-time offer.
-->

---

## A Few Honest Pieces of Advice

- **You don't need to know everything** — nobody does. Be comfortable saying "I don't know yet, but I can find out."
- **Depth beats breadth early on** — go deep on one project rather than dabbling in ten.
- **Fundamentals > frameworks** — tools change every year; principles don't.
- **Curiosity and consistency** win over raw talent.

<!--
Speak from experience — a personal "learned it the hard way" anecdote is very effective.
Reassure them: imposter syndrome is normal; everyone in industry googles things daily.
-->

---

## Key Takeaways

- Industry AI is **real, valuable, and mostly invisible** — it cuts cost and grows revenue.
- The ML engineer's job is **~80% engineering**, not modeling.
- **Software fundamentals + growth mindset + communication** matter more than any single tool.
- Start now: **build, deploy, and share** end-to-end projects.

<!--
Keep this tight — this is the message they carry out the door.
-->

---

<!-- _class: lead invert -->

## Q & A

What questions do you have?

*Day-to-day work · how to break in · specific tech · anything about the industry*

<!--
If it's quiet, prompt them: "A common question is HOW DID YOU GET YOUR FIRST ML ROLE?
— want me to answer that?" Offer LinkedIn / email for follow-ups.
-->

---

<!-- _class: lead invert -->

# Thank You

**Setia Budi**
Machine Learning Engineer, tiket.com

*Now go build something.*

<!--
End on the call to action: encourage them to start a project THIS WEEK.
-->
