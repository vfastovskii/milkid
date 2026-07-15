# AGENTS

<skills_system priority="1">

## Available Skills

<!-- SKILLS_TABLE_START -->
<usage>
When users ask you to perform tasks, check if any of the available skills below can help complete the task more effectively. Skills provide specialized capabilities and domain knowledge.

How to use skills:
- Invoke: `npx openskills read <skill-name>` (run in your shell)
  - For multiple: `npx openskills read skill-one,skill-two`
- The skill content will load with detailed instructions on how to complete the task
- Base directory provided in output for resolving bundled resources (references/, scripts/, assets/)

Usage notes:
- Only use skills listed in <available_skills> below
- Do not invoke a skill that is already loaded in your context
- Each skill invocation is stateless
</usage>

<available_skills>

<skill>
<name>anti-slop</name>
<description>Comprehensive toolkit for detecting and eliminating "AI slop" - generic, low-quality AI-generated patterns in natural language, code, and design. Use when reviewing or improving content quality, preventing generic AI patterns, cleaning up existing content, or enforcing quality standards in writing, code, or design work.</description>
<location>project</location>
</skill>

<skill>
<name>discover-agentic</name>
<description>Automatically discover agentic workflow skills when building AI agents, implementing tool use patterns, managing context windows, decomposing complex tasks, or designing multi-step autonomous workflows. Activates for agentic AI development.</description>
<location>project</location>
</skill>

<skill>
<name>discover-api</name>
<description>Automatically discover API design skills when working with REST APIs, GraphQL schemas, API authentication, OAuth, JWT, rate limiting, API versioning, error handling, or endpoint design. Activates for backend API development tasks.</description>
<location>project</location>
</skill>

<skill>
<name>discover-cicd</name>
<description>Automatically discover CI/CD and automation skills when working with GitHub Actions, Jenkins, GitLab CI, pipelines, continuous integration, continuous deployment, or automated testing. Activates for CI/CD development tasks.</description>
<location>project</location>
</skill>

<skill>
<name>discover-cryptography</name>
<description>Automatically discover cryptography skills when working with encryption, TLS, certificates, PKI, and security</description>
<location>project</location>
</skill>

<skill>
<name>discover-data</name>
<description>Automatically discover data pipeline and ETL skills when working with ETL, data pipelines, streaming, batch processing, data validation, or pipeline orchestration. Activates for data development tasks.</description>
<location>project</location>
</skill>

<skill>
<name>discover-database</name>
<description>Automatically discover database skills when working with SQL, PostgreSQL, MongoDB, Redis, database schema design, query optimization, migrations, connection pooling, ORMs, or database selection. Activates for database design, optimization, and implementation tasks.</description>
<location>project</location>
</skill>

<skill>
<name>discover-debugging</name>
<description>Automatically discover debugging and profiling skills when working with GDB, LLDB, breakpoints, profiling, stack traces, memory leaks, core dumps, or performance profiling. Activates for debugging development tasks.</description>
<location>project</location>
</skill>

<skill>
<name>discover-distributed</name>
<description>Automatically discover distributed systems and realtime communication skills when working with consensus, CRDTs, replication, WebSocket, SSE, pub/sub, or event-driven architectures</description>
<location>project</location>
</skill>

<skill>
<name>discover-engineering</name>
<description>Automatically discover software engineering practice skills when working with code review, documentation, pair programming, production debugging, performance profiling, deployment strategies, or software engineering practices. Activates for engineering development tasks.</description>
<location>project</location>
</skill>

<skill>
<name>discover-frontend</name>
<description>Automatically discover frontend development skills when working with React, Next.js, UI components, state management, data fetching, forms, accessibility, performance optimization, or SEO. Activates for frontend web development tasks.</description>
<location>project</location>
</skill>

<skill>
<name>discover-infra</name>
<description>Automatically discover cloud, infrastructure, deployment, and container skills when working with AWS, GCP, Azure, Docker, Kubernetes, Terraform, Netlify, Heroku, serverless, or IaC</description>
<location>project</location>
</skill>

<skill>
<name>discover-math</name>
<description>Automatically discover mathematics and algorithm skills when working with linear algebra, calculus, optimization, statistics, probability, numerical methods, category theory, or topology. Activates for math development tasks.</description>
<location>project</location>
</skill>

<skill>
<name>discover-mcp</name>
<description>Automatically discover MCP (Model Context Protocol) skills when building MCP servers, designing tools, implementing resources/prompts, or testing MCP integrations. Activates for MCP server development tasks.</description>
<location>project</location>
</skill>

<skill>
<name>discover-ml</name>
<description>Automatically discover machine learning and AI skills when working with machine learning, PyTorch, training, inference, RAG, embeddings, fine-tuning, LLM, DSPy, HuggingFace, or diffusion models. Activates for ML development tasks.</description>
<location>project</location>
</skill>

<skill>
<name>discover-mobile</name>
<description>Automatically discover mobile development skills when working with iOS, Android, Swift, SwiftUI, React Native, mobile development, SwiftData, or app development. Activates for mobile development tasks.</description>
<location>project</location>
</skill>

<skill>
<name>discover-networking</name>
<description>Automatically discover networking and connectivity skills when working with TCP, UDP, DNS, mTLS, NAT traversal, SSH, Mosh, Tailscale, or network resilience. Activates for networking development tasks.</description>
<location>project</location>
</skill>

<skill>
<name>discover-product</name>
<description>Automatically discover product management skills when working with product management, roadmap, user stories, prioritization, metrics, or product strategy. Activates for product development tasks.</description>
<location>project</location>
</skill>

<skill>
<name>discover-research</name>
<description>Automatically discover research methodology skills when working with research methodology, literature review, systematic review, evidence synthesis, academic research, or experimental design. Activates for research tasks.</description>
<location>project</location>
</skill>

<skill>
<name>discover-security</name>
<description>Automatically discover security skills when working with authentication, authorization, input validation, security headers, vulnerability assessment, or secrets management. Activates for application security, OWASP, and security hardening tasks.</description>
<location>project</location>
</skill>

<skill>
<name>discover-systems-theory</name>
<description>Automatically discover eBPF, compiler, programming language theory, information retrieval, and formal verification skills when working with kernel tracing, parsers, type systems, Z3, Lean, or theorem proving</description>
<location>project</location>
</skill>

<skill>
<name>discover-testing</name>
<description>Automatically discover testing skills when working with unit testing, integration testing, e2e testing, TDD, test coverage, mocking, pytest, Jest, or test automation. Activates for testing development tasks.</description>
<location>project</location>
</skill>

<skill>
<name>discover-wasm</name>
<description>Automatically discover WebAssembly skills when working with WebAssembly, WASM, WASI, wasm-bindgen, Rust to WASM, wasm-pack, or browser runtime. Activates for WASM development tasks.</description>
<location>project</location>
</skill>

<skill>
<name>discover-zig</name>
<description>Automatically discover Zig programming skills when working with Zig, comptime, allocators, build.zig, safety, C interop, memory management, or systems programming. Activates for Zig development tasks.</description>
<location>project</location>
</skill>

<skill>
<name>elegant-design</name>
<description>Create world-class, accessible, responsive interfaces with sophisticated interactive elements including chat, terminals, code display, and streaming content. Use when building user interfaces that need professional polish and developer-focused features.</description>
<location>project</location>
</skill>

<skill>
<name>typed-holes-refactor</name>
<description>Refactor codebases using Design by Typed Holes methodology - iterative, test-driven refactoring with formal hole resolution, constraint propagation, and continuous validation. Use when refactoring existing code, optimizing architecture, or consolidating technical debt through systematic hole-driven development.</description>
<location>project</location>
</skill>

</available_skills>
<!-- SKILLS_TABLE_END -->

</skills_system>
