# NG12 Assessor Architecture Diagrams

## 1. PDF Ingestion Pipeline

```mermaid
flowchart TB
    subgraph Input["Input"]
        PDF["NG12 PDF<br/>(85+ pages)"]
    end

    subgraph Parsing["PDF Parsing"]
        PyMuPDF["PyMuPDF Parser"]
        Lines["Line-by-Line Extraction"]
        PDF --> PyMuPDF --> Lines
    end

    subgraph StateMachine["State Machine<br/>(Text Pattern Detection)"]
        SM["Pattern-Based<br/>State Transitions"]
        PartA["PART_A<br/>Pages 8-36<br/>Clinical Recommendations<br/>(section 1.x / 1.x.y)"]
        PartB["PART_B<br/>Pages 37-82<br/>Triggered by:<br/>'recommendations organised<br/>by symptom'"]
        Stop["STOP<br/>Pages 83+<br/>Triggered by markers:<br/>'terms used in this guideline'<br/>'rationale and impact' etc."]
        Lines --> SM
        SM --> PartA
        SM --> PartB
        SM --> Stop
    end

    subgraph PartAProcess["Part A Processing<br/>(Clinical Recommendations)"]
        RuleDetect["Rule Detection<br/>RE_SUBSECTION: 1.x.y Pattern"]
        RuleChunk["Rule Chunking"]
        MetaExtract["Metadata Extraction<br/>(extract_rule_metadata)"]
        PartA --> RuleDetect --> RuleChunk --> MetaExtract
    end

    subgraph PartBProcess["Part B Processing<br/>(Symptom Index Tables)"]
        TableParse["Table Row Detection"]
        SymptomChunk["Symptom Chunking"]
        RefExtract["Reference Extraction<br/>[1.x.y]"]
        PartB --> TableParse --> SymptomChunk --> RefExtract
    end

    subgraph Embedding["Embedding Generation"]
        Vertex["Vertex AI<br/>text-embedding-004"]
    end

    subgraph Collections["ChromaDB Collections"]
        C1["<b>ng12_canonical</b><br/>---<br/>Verbatim rule text<br/>ID-based lookup<br/>Citation display<br/>~100-150 entries"]
        C2["<b>ng12_search</b><br/>---<br/>Enriched text<br/>Vector similarity<br/>Synonym expansion<br/>Primary retrieval"]
        C3["<b>ng12_symptom</b><br/>---<br/>Symptom to Cancer mapping<br/>Part B table rows<br/>Cross-references<br/>~150-300 entries"]
    end

    MetaExtract --> |"rule_canonical<br/>(no embedding)"| C1
    MetaExtract --> |"rule_search<br/>+ embeddings"| Vertex --> C2
    RefExtract --> |"symptom_index<br/>+ embeddings"| Vertex --> C3

    style C1 fill:#e1f5fe,stroke:#0288d1
    style C2 fill:#fff3e0,stroke:#f57c00
    style C3 fill:#f3e5f5,stroke:#7b1fa2
```

### Collection Details

| Collection | Contents | Purpose | Query Method |
|-----------|---------|------|---------|
| **ng12_canonical** | Original rule text with full metadata | Citation display, Admin page | ID lookup (`get_canonical(rule_id)`) |
| **ng12_search** | Enriched text + synonym expansion | Primary semantic retrieval | Vector similarity |
| **ng12_symptom** | Symptom-to-cancer mapping (Part B table rows) | Symptom index cross-references | Vector + metadata filter |

### Chunk Metadata by Type

| Field | rule_canonical | rule_search | symptom_index |
|-------|:-:|:-:|:-:|
| `section` | Y | Y | - |
| `action_type` | Y | - | - |
| `cancer_type` | Y | Y | - |
| `page` / `page_end` | Y | Y | Y |
| `age_min` / `age_max` | Y | - | - |
| `symptom_keywords_json` | Y | - | - |
| `risk_factor_smoking` | Y | - | - |
| `urgency` | Y | - | - |
| `gender_specific` | Y | - | - |
| `system_title` / `sub_title` | - | - | Y |
| `symptom` / `possible_cancer` | - | - | Y |
| `references_json` | - | - | Y |
| `rule_id` | - | Y | - |

---

## 2. Patient Assessment Workflow

```mermaid
flowchart TB
    subgraph Input["API Input"]
        API["POST /assess/{patient_id}"]
    end

    subgraph FetchPatient["Node: fetch_patient"]
        Gemini1["Gemini 2.0 Flash"]
        FuncDef["Function Definition:<br/>get_patient_data(patient_id)"]
        DirectFallback["Direct Fallback:<br/>patient_db.get_patient()"]

        API --> Gemini1
        Gemini1 --> |"function_call<br/>(if available)"| FuncDef
        Gemini1 -.-> |"fallback<br/>(no credentials)"| DirectFallback
    end

    subgraph PatientDB["Patient Data"]
        JSON["patients.json<br/>(10 test patients)"]
        PatientData["Patient Record:<br/>age, symptoms,<br/>smoking_history,<br/>gender, duration"]
        FuncDef --> JSON --> PatientData
        DirectFallback --> JSON
    end

    subgraph RAG["Node: retrieve_guidelines"]
        QueryBuild["Query Construction:<br/>'symptoms age gender smoking_history'"]

        VectorSearch["vector_store.query()<br/>on ng12_guidelines"]
        Rerank["Patient-Based Reranking<br/>(rag_pipeline.retrieve)"]

        Chunks["Retrieved Chunks<br/>(top_k=8)"]

        PatientData --> QueryBuild --> VectorSearch
        VectorSearch --> Rerank --> Chunks
    end

    subgraph Reasoning["Node: assess_risk"]
        Gemini2["Gemini 2.0 Flash<br/>temperature=0.1<br/>max_tokens=2048"]
        SystemPrompt["ASSESSMENT_SYSTEM_PROMPT:<br/>Match patient to criteria<br/>Check ALL conditions<br/>Respond in JSON only"]
        Assessment["JSON Response:<br/>risk_level<br/>cancer_type<br/>recommended_action<br/>reasoning<br/>matched_recommendations"]

        Chunks --> Gemini2
        PatientData --> Gemini2
        SystemPrompt --> Gemini2
        Gemini2 --> Assessment
    end

    subgraph Output["API Response"]
        Citation["Citations from<br/>chunk metadata"]
        Response["AssessResponse JSON"]
        Assessment --> Citation --> Response
    end

    style Gemini1 fill:#bbdefb,stroke:#1976d2
    style Gemini2 fill:#bbdefb,stroke:#1976d2
    style Chunks fill:#fff3e0,stroke:#f57c00
    style Assessment fill:#c8e6c9,stroke:#388e3c
```

### LangGraph Workflow Nodes

```mermaid
stateDiagram-v2
    [*] --> fetch_patient
    fetch_patient --> handle_error: error
    fetch_patient --> retrieve_guidelines: patient data
    retrieve_guidelines --> handle_error: error
    retrieve_guidelines --> assess_risk: chunks
    assess_risk --> [*]
    handle_error --> [*]

    note right of fetch_patient
        Tries Gemini function calling
        Falls back to direct DB lookup
    end note

    note right of retrieve_guidelines
        rag_pipeline.retrieve()
        top_k=8, patient_data=patient
    end note

    note right of assess_risk
        Gemini 2.0 Flash
        Returns structured JSON
    end note
```

---

## 3. Chat Workflow (LangGraph)

```mermaid
flowchart TB
    subgraph Entry["Entry Point"]
        API["POST /chat<br/>{session_id, message}"]
    end

    subgraph Node1["Node: load_history"]
        LoadHist["session_store.get_history()"]
        History["Conversation History<br/>+ Current Topic"]
    end

    subgraph Node1b["Node: input_guardrail"]
        Classify["classify_input(message)"]
        Smalltalk["smalltalk<br/>(greetings, thanks)"]
        Meta["meta<br/>(who are you, help)"]
        MedOOS["medical_out_of_scope<br/>(treatment, prognosis)"]
        Proceed["proceed<br/>(NG12 question)"]
    end

    subgraph NodeSM["Node: smalltalk_meta"]
        SmallResp["SMALLTALK_RESPONSE<br/>or META_RESPONSE"]
    end

    subgraph Node2["Node: build_query"]
        QueryBuilder["QueryBuilder.build()"]
        Strategy["Strategy Decision"]
        Direct["Direct<br/>(standalone question)"]
        Topic["Topic Enriched<br/>(follow-up)"]
        LLMRewrite["LLM Rewrite<br/>(fallback)"]

        QueryBuilder --> Strategy
        Strategy --> Direct
        Strategy --> Topic
        Strategy --> LLMRewrite
    end

    subgraph Node3["Node: retrieve"]
        RAG["rag_pipeline.retrieve()"]
        Chunks["Retrieved Chunks<br/>with scores (top_k=6)"]
    end

    subgraph Node4["Node: guardrail_check"]
        ScoreCheck["_assess_chunk_quality()"]
        ScopeCheck["Out-of-Scope Check"]

        Sufficient["sufficient"]
        Weak["weak"]
        None_["none"]
        OutOfScope["out_of_scope"]
    end

    subgraph NodeSQ["Node: summarize_query"]
        SumQ["Extract structured<br/>clinical info<br/>(display-only metadata)"]
    end

    subgraph ConditionalRouting["Conditional Routing"]
        Router{{"route_guardrail()"}}
    end

    subgraph Node5a["Node: generate"]
        GenPrompt["format_chat_prompt()"]
        GenLLM["Gemini 2.0 Flash"]
        GenAnswer["Full Answer<br/>+ Citations"]
    end

    subgraph Node5b["Node: qualify"]
        QualPrompt["Generate partial answer"]
        QualWrap["CHAT_QUALIFY_TEMPLATE"]
        QualAnswer["Qualified Answer"]
    end

    subgraph Node5c["Node: refuse"]
        RefuseMsg["CHAT_REFUSE_RESPONSE"]
        NoCite["No Citations"]
    end

    subgraph Node5d["Node: out_of_scope"]
        OOSMsg["CHAT_OUT_OF_SCOPE_RESPONSE"]
        OOSCite["No Citations"]
    end

    subgraph Node6["Node: save_history"]
        SaveHist["session_store.append()"]
        UpdateTopic["session_store.update_topic()<br/>(only if sufficient/weak)"]
    end

    subgraph Output["Output"]
        Response["ChatResponse JSON:<br/>answer<br/>citations<br/>session_id"]
    end

    API --> Node1
    Node1 --> Node1b

    Classify --> Smalltalk
    Classify --> Meta
    Classify --> MedOOS
    Classify --> Proceed

    Smalltalk --> NodeSM
    Meta --> NodeSM
    MedOOS --> Node5d
    Proceed --> Node2

    NodeSM --> Node6

    Node2 --> Node3
    Node3 --> Node4
    Node4 --> ConditionalRouting

    Router --> |"sufficient"| NodeSQ
    Router --> |"weak"| NodeSQ
    Router --> |"none"| Node5c
    Router --> |"out_of_scope"| Node5d

    NodeSQ --> |"sufficient"| Node5a
    NodeSQ --> |"weak"| Node5b

    Node5a --> Node6
    Node5b --> Node6
    Node5c --> Node6
    Node5d --> Node6

    Node6 --> Output

    style Node4 fill:#fff9c4,stroke:#f9a825
    style ConditionalRouting fill:#ffcdd2,stroke:#d32f2f
    style Node5a fill:#c8e6c9,stroke:#388e3c
    style Node5b fill:#fff3e0,stroke:#f57c00
    style Node5c fill:#ffcdd2,stroke:#d32f2f
    style Node5d fill:#e1bee7,stroke:#7b1fa2
    style Node1b fill:#fff9c4,stroke:#f9a825
```

---

## 4. Retrieval Pipeline (Detailed)

```mermaid
flowchart TB
    subgraph EntryA["Assessment Entry<br/>POST /assess/{patient_id}"]
        PatientID["patient_id"]
        FetchPat["fetch_patient()"]
        PatientData["patient_data:<br/>age, symptoms,<br/>smoking_history, gender"]
        QueryBuildA["Query Construction:<br/>f'{symptoms} age {age}<br/>{gender} {smoking_history}'"]

        PatientID --> FetchPat --> PatientData --> QueryBuildA
    end

    subgraph EntryB["Chat Entry<br/>POST /chat"]
        UserMsg["user message"]
        QB["QueryBuilder.build()<br/>(A+C+B tiers)"]
        SearchQuery["search_query"]

        UserMsg --> QB --> SearchQuery
    end

    subgraph Retrieve["retrieve(query, top_k, patient_data)"]
        FetchK["fetch_k = top_k x 3<br/>(assessment: 24, chat: 18)"]

        subgraph VectorQuery["vector_store.query(query, fetch_k)"]
            SearchColl["ng12_search"]
            SymColl["ng12_symptom"]
            VS["Cosine Similarity<br/>score = 1 - distance"]
        end

        CandidatePool["Candidate Pool<br/>(mixed rule_search +<br/>symptom_index chunks)"]

        FetchK --> VectorQuery
        SearchColl --> VS
        SymColl --> VS
        VS --> CandidatePool
    end

    QueryBuildA --> |"top_k=8<br/>patient_data=patient"| FetchK
    SearchQuery --> |"top_k=6<br/>patient_data=None"| FetchK

    subgraph ModeCheck["Mode Branch"]
        HasPatient{{"patient_data?"}}
    end

    CandidatePool --> HasPatient

    subgraph AssessRerank["Assessment Reranking<br/>(deterministic patient-based)"]
        direction TB
        LoopA["For each chunk in candidates:"]
        AB1["+0.15 if patient_age >= age_min"]
        AB2["+0.15 if patient_age < age_max"]
        AB3["+0.10 per symptom keyword overlap<br/>(fuzzy: cs in ps or ps in cs)"]
        AB4["+0.10 if smoker + risk_factor_smoking"]
        AB5["+0.05 if gender matches<br/>-0.30 if gender clashes"]
        AScore["result.score += boost"]

        LoopA --> AB1 --> AB2 --> AB3 --> AB4 --> AB5 --> AScore
    end

    subgraph ChatRerank["Chat Reranking<br/>_chat_rerank(query, results)"]
        direction TB
        Intent["Detect query intent:<br/>q_urgency (urgent|red flag|emergency)<br/>q_age (age|under N|over N|years old)<br/>q_duration (weeks|months|persistent)<br/>q_exact (quote|exact|verbatim)"]
        LoopC["For each chunk in candidates:"]
        CB1["+0.10 if q_urgency AND<br/>chunk.urgency in<br/>{immediate, very_urgent, urgent}"]
        CB2["+0.10 if q_age AND<br/>chunk has age_min or age_max"]
        CB3["+0.10 if q_duration AND<br/>chunk text matches duration regex"]
        CB4["+0.15 if q_exact AND<br/>doc_type == 'rule_search'"]
        CScore["result.score += boost"]

        Intent --> LoopC --> CB1 --> CB2 --> CB3 --> CB4 --> CScore
    end

    HasPatient --> |"Yes"| AssessRerank
    HasPatient --> |"No"| ChatRerank

    subgraph Select["Sort and Select"]
        SortDesc["results.sort(score, descending)"]
        TopK["results = results[:top_k]<br/>(assessment: top 8, chat: top 6)"]
        SortDesc --> TopK
    end

    AScore --> SortDesc
    CScore --> SortDesc

    subgraph Canonical["_attach_canonicals(results)"]
        CanonicalColl["ng12_canonical<br/>Collection"]

        subgraph RuleAttach["For each rule_search chunk"]
            RuleLookup["get_canonical(rule_id)<br/>e.g. rule_id='1.1.1'"]
            RuleResult["Attach:<br/>canonical_text<br/>canonical_metadata"]
            RuleLookup --> RuleResult
        end

        subgraph SymAttach["For each symptom_index chunk"]
            RefParse["Parse references_json<br/>e.g. ['[1.5.2]', '[1.1.1]']"]
            RefStrip["Strip brackets<br/>'[1.5.2]' -> '1.5.2'"]
            RefLookup["get_canonical(rule_id)<br/>for each reference"]
            SymResult["Attach:<br/>referenced_canonicals[]<br/>{rule_id, text, metadata}"]
            RefParse --> RefStrip --> RefLookup --> SymResult
        end

        CanonicalColl --> RuleLookup
        CanonicalColl --> RefLookup
    end

    TopK --> RuleAttach
    TopK --> SymAttach

    subgraph Output["Output"]
        Results["Retrieved Chunks list:<br/>chunk_id, text, metadata,<br/>score, canonical_text,<br/>canonical_metadata,<br/>referenced_canonicals"]
    end

    RuleResult --> Results
    SymResult --> Results

    style EntryA fill:#e1f5fe,stroke:#0288d1
    style EntryB fill:#e8f5e9,stroke:#43a047
    style VectorQuery fill:#fff3e0,stroke:#f57c00
    style AssessRerank fill:#fff3e0,stroke:#f57c00
    style ChatRerank fill:#e8f5e9,stroke:#43a047
    style Canonical fill:#f3e5f5,stroke:#7b1fa2
```

### Score Calculation Details

```
final_score = base_vector_similarity + boost

=== Assessment Mode (patient_data provided) ===

┌──────────────────────────────────────────────────────────┐
│  Boost Type          │ Condition                │ Value   │
├──────────────────────────────────────────────────────────┤
│  Age Min Match       │ patient_age >= age_min    │ +0.15  │
│  Age Max Match       │ patient_age < age_max     │ +0.15  │
│  Symptom Overlap     │ per overlapping keyword   │ +0.10  │
│  Smoking Match       │ smoker + risk_factor=True │ +0.10  │
│  Gender Match        │ same gender               │ +0.05  │
│  Gender Clash        │ opposite gender           │ -0.30  │
└──────────────────────────────────────────────────────────┘

=== Chat Mode (no patient_data) ===

┌──────────────────────────────────────────────────────────┐
│  Boost Type          │ Condition                │ Value   │
├──────────────────────────────────────────────────────────┤
│  Urgency Match       │ query has urgent/red flag │ +0.10  │
│  Age Query           │ query mentions age/years  │ +0.10  │
│  Duration Query      │ query has weeks/months    │ +0.10  │
│  Exact Wording       │ query has quote/verbatim  │ +0.15  │
└──────────────────────────────────────────────────────────┘

Example (Assessment Mode):
Query: "55 year old male with hemoptysis"
Patient: age=55, symptoms=["hemoptysis"], smoking="Current Smoker"

Chunk: [1.1.1] Lung cancer urgent referral for haemoptysis, 40+
  - Vector similarity: 0.72
  - Age min match (55 >= 40): +0.15
  - Symptom overlap (hemoptysis): +0.10
  - Smoking match: +0.10
  - Gender match (Male): +0.05
  ─────────────────────────────
  Final Score: 1.12
```

---

## 5. Chat Detail Diagrams

### 5.1 Memory & Topic Management

```mermaid
flowchart TB
    subgraph SessionStore["SessionStore (In-Memory)"]
        Structure["_sessions: dict[str, list]<br/>_topics: dict[str, str]"]
    end

    subgraph History["History Management"]
        Append["append(session_id, role, content)"]
        GetHist["get_history(session_id)"]
        Clear["clear_session(session_id)"]

        HistoryFormat["History Format:<br/>[<br/>  {role: 'user', content: '...'},<br/>  {role: 'assistant', content: '...'},<br/>  ...<br/>]"]
    end

    subgraph TopicMgmt["Topic Management"]
        UpdateTopic["update_topic(session_id, chunks)"]
        GetTopic["get_topic(session_id)"]

        TopicExtract["Topic Extraction:<br/>1. Filter out non-cancer chunks<br/>2. Most common cancer_type<br/>3. Up to 2 clinical terms<br/>(section numbers excluded<br/>to avoid search noise)"]

        TopicFormat["Topic Format:<br/>'lung hemoptysis dysphagia'"]
    end

    subgraph Usage["Usage in Query Building"]
        FollowUp["Follow-up Detection"]
        Enrich["Topic Enrichment"]

        Example1["Q1: 'lung cancer referral criteria'<br/>topic = 'lung hemoptysis'"]
        Example2["Q2: 'what about under 40?'<br/>query = 'lung hemoptysis<br/>what about under 40?'"]
    end

    subgraph UpdateRule["Topic Update Rules"]
        Rule1["Only updates when:<br/>guardrail_result = sufficient/weak<br/>AND citations exist"]
        Rule2["Uses only cited chunks<br/>(matched by chunk_id)"]
    end

    Structure --> History
    Structure --> TopicMgmt
    TopicMgmt --> Usage
    TopicMgmt --> UpdateRule

    style TopicExtract fill:#e8f5e9,stroke:#43a047
    style Example2 fill:#fff3e0,stroke:#f57c00
    style UpdateRule fill:#fff9c4,stroke:#f9a825
```

### 5.2 Query Builder Strategy (A+C+B Tiers)

```mermaid
flowchart TB
    subgraph Input["Input"]
        Message["User Message"]
        SessionID["Session ID"]
    end

    subgraph Detection["Follow-up Detection<br/>(is_followup)"]
        Checks["Checks (ANY true = follow-up):<br/>1. len(words) <= 3 (very short)<br/>2. starts with known phrase<br/>(what about, how about,<br/>and if, what if, etc.)<br/>3. len(words) < 8 AND<br/>has context pronoun<br/>(it, that, they, this, them)"]
        IsFollowup{{"is_followup?"}}
    end

    subgraph StrategyA["Tier A: Direct"]
        DirectQuery["query = message"]
        DirectLabel["strategy = 'direct'"]
        DirectUse["For: standalone questions"]
    end

    subgraph StrategyC["Tier C: Topic Enriched"]
        GetTopic2["topic = session_store.get_topic()"]
        HasTopic{{"topic exists?"}}
        EnrichQuery["query = f'{topic} {message}'"]
        EnrichLabel["strategy = 'topic_enriched'"]
        EnrichUse["For: follow-up questions"]
    end

    subgraph StrategyB["Tier B: LLM Rewrite"]
        HasGemini{{"gemini available?"}}
        GetHistory["history = get_history()<br/>(max_turns=6)"]
        RewritePrompt["REWRITE_PROMPT:<br/>'Rewrite into standalone<br/>search query for NG12'"]
        LLMCall["Gemini 2.0 Flash"]
        RewriteQuery["query = rewritten"]
        RewriteLabel["strategy = 'llm_rewrite'"]
        RewriteUse["For: fallback"]
    end

    subgraph Output["Output"]
        Result["(query, strategy)"]
    end

    Message --> Detection
    SessionID --> Detection
    Detection --> IsFollowup

    IsFollowup --> |"No"| StrategyA --> Result
    IsFollowup --> |"Yes"| StrategyC

    StrategyC --> HasTopic
    HasTopic --> |"Yes"| EnrichQuery --> Result
    HasTopic --> |"No"| StrategyB

    StrategyB --> HasGemini
    HasGemini --> |"Yes"| GetHistory --> RewritePrompt --> LLMCall --> RewriteQuery --> Result
    HasGemini --> |"No"| DirectQuery

    style StrategyA fill:#c8e6c9,stroke:#388e3c
    style StrategyC fill:#fff3e0,stroke:#f57c00
    style StrategyB fill:#bbdefb,stroke:#1976d2
```

### 5.3 Query Rewriting / Summarization

```mermaid
flowchart LR
    subgraph Context["Context"]
        History["History:<br/>U: lung cancer referral?<br/>A: For patients 40+ with...<br/>U: what about smokers?<br/>A: Smoking history is a..."]
        Current["Current: 'and under 40?'"]
    end

    subgraph Rewrite["LLM Rewrite"]
        Prompt["REWRITE_PROMPT:<br/>'Rewrite into standalone<br/>search query for NG12<br/>Under 20 words<br/>Keep medical terms exact'"]
        Gemini["Gemini 2.0 Flash"]
        Rewritten["'lung cancer urgent referral<br/>criteria for patients under 40<br/>years old'"]
    end

    subgraph Usage["Used In"]
        QB["QueryBuilder Tier B<br/>(build_query_node)"]
        GR["Guardrail Retry<br/>(guardrail_check_node:<br/>when result='none' and<br/>strategy != 'llm_rewrite')"]
    end

    Context --> Prompt --> Gemini --> Rewritten
    Rewritten --> QB
    Rewritten --> GR

    style Rewritten fill:#e8f5e9,stroke:#43a047
```

### 5.4 Input/Output Guardrails

```mermaid
flowchart TB
    subgraph InputGuard["Input Guardrails (classify_input)"]
        InputCheck["Input Classification"]

        subgraph SmallMeta["Smalltalk / Meta"]
            STalk["Greetings, thanks, bye"]
            MetaQ["Who are you, help,<br/>what can you do"]
        end

        subgraph InScope["In-Scope (proceed)"]
            Medical["Referral questions"]
            Symptom["Symptom queries"]
            Age["Age threshold questions"]
            Urgent["Urgency criteria"]
        end

        subgraph MedOOS["Medical Out-of-Scope"]
            Treatment["chemotherapy, radiotherapy,<br/>surgery, medication, drug"]
            Prognosis["prognosis, survival rate,<br/>life expectancy, mortality"]
            Diagnosis["do i have cancer,<br/>diagnose me, is this cancer"]
            Override["OVERRIDE: if message also<br/>contains referral/criteria/<br/>symptom keywords -> proceed"]
        end

        InputCheck --> SmallMeta
        InputCheck --> InScope
        InputCheck --> MedOOS
    end

    subgraph OutputGuard["Output Guardrails (_assess_chunk_quality)"]
        ScoreAnalysis["Retrieved Chunks Analysis"]

        subgraph Thresholds["Quality Assessment Logic"]
            Step1["IF all scores < 0.25 -> none"]
            Step2["Count good_chunks (score > 0.4)"]
            Step3["IF good_chunks=0 AND best < 0.35<br/>-> none"]
            Step4["IF good_chunks=0 AND best >= 0.35<br/>-> weak"]
            Step5["IF good_chunks <= 2 AND best < 0.5<br/>-> weak"]
            Step6["Otherwise -> sufficient"]
        end

        ScoreAnalysis --> Thresholds
    end

    subgraph OOSCheck["Post-Retrieval OOS Check"]
        Keywords["_OUT_OF_SCOPE_KEYWORDS:<br/>treatment, chemotherapy,<br/>prognosis, survival rate,<br/>staging, metastasis,<br/>palliative, immunotherapy..."]
        InScopeKW["_IN_SCOPE_KEYWORDS override:<br/>referral, investigation,<br/>symptom, criteria, ng12..."]
    end

    subgraph Routing["Response Routing"]
        Route{{"guardrail_result"}}

        GenNode["generate<br/>Full answer + citations"]
        QualNode["qualify<br/>Partial answer + caveat"]
        RefuseNode["refuse<br/>Cannot answer message"]
        OOSNode["out_of_scope<br/>Outside NG12 scope"]
        SmNode["smalltalk_meta<br/>Canned response"]
    end

    SmallMeta --> |"smalltalk/meta"| SmNode
    MedOOS --> |"medical_out_of_scope"| OOSNode
    InScope --> OutputGuard
    OutputGuard --> OOSCheck
    OOSCheck --> Route

    Route --> |"sufficient"| GenNode
    Route --> |"weak"| QualNode
    Route --> |"none"| RefuseNode
    Route --> |"out_of_scope"| OOSNode

    style GenNode fill:#c8e6c9,stroke:#388e3c
    style QualNode fill:#fff3e0,stroke:#f57c00
    style RefuseNode fill:#ffcdd2,stroke:#d32f2f
    style OOSNode fill:#e1bee7,stroke:#7b1fa2
    style SmNode fill:#bbdefb,stroke:#1976d2
```

### Guardrail Response Templates

```
┌──────────────────────────────────────────────────────────────────┐
│ sufficient -> Full Response (generate_node)                      │
├──────────────────────────────────────────────────────────────────┤
│ {LLM generated answer with [Source N] citations}                 │
│                                                                   │
│ Sources (cleaned by clean_answer_sources):                        │
│ [NG12 §1.1.1, p.9] - Lung cancer referral criteria              │
│ [NG12 §1.1.2, p.9] - Urgent chest X-ray                         │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ weak -> Qualified Response (CHAT_QUALIFY_TEMPLATE)               │
├──────────────────────────────────────────────────────────────────┤
│ Based on the limited evidence found in the NG12 guidelines,      │
│ I can share the following, but please note this may not fully    │
│ address your question:                                            │
│                                                                   │
│ {partial_answer}                                                  │
│                                                                   │
│ For a more complete answer, you may want to ask about a          │
│ specific cancer type, symptom, or referral pathway.              │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ none -> Refuse Response (CHAT_REFUSE_RESPONSE)                   │
├──────────────────────────────────────────────────────────────────┤
│ I don't have sufficient evidence in the NG12 guidelines to       │
│ answer this question. The retrieved passages don't appear to     │
│ be relevant enough to provide a grounded response.               │
│                                                                   │
│ Could you try:                                                    │
│ - Asking about a specific cancer type (e.g., lung, breast)       │
│ - Asking about a specific symptom (e.g., haemoptysis)            │
│ - Asking about referral criteria for a particular age group      │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ out_of_scope -> Out-of-Scope Response (CHAT_OUT_OF_SCOPE_...)    │
├──────────────────────────────────────────────────────────────────┤
│ This question appears to fall outside the scope of the NG12      │
│ Suspected Cancer: Recognition and Referral guideline.            │
│                                                                   │
│ NG12 covers criteria for referring patients with suspected       │
│ cancer symptoms for urgent investigation or specialist           │
│ assessment.                                                       │
│                                                                   │
│ I can help with questions about:                                  │
│ - Which symptoms trigger urgent referral for specific cancers    │
│ - Age thresholds and risk factors for referral criteria          │
│ - The difference between urgent referral and investigation       │
│ - Safety netting recommendations                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ smalltalk -> Greeting Response (SMALLTALK_RESPONSE)              │
├──────────────────────────────────────────────────────────────────┤
│ Hello! I'm a clinical guidelines assistant specializing in       │
│ the NICE NG12 guideline for suspected cancer recognition         │
│ and referral.                                                     │
│                                                                   │
│ I can help you with:                                              │
│ - Referral criteria for specific cancer types                     │
│ - Age thresholds and risk factors                                 │
│ - Understanding urgent referral vs urgent investigation           │
│ - Safety netting recommendations                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ meta -> System Info Response (META_RESPONSE)                      │
├──────────────────────────────────────────────────────────────────┤
│ I'm a clinical decision support assistant that answers questions │
│ about the NICE NG12 guideline: Suspected Cancer - Recognition    │
│ and Referral.                                                     │
│                                                                   │
│ I use RAG to find relevant guideline passages and provide        │
│ grounded answers with citations. I only answer based on the      │
│ NG12 guideline content.                                           │
└──────────────────────────────────────────────────────────────────┘
```

---

## System Architecture Overview

```mermaid
flowchart TB
    subgraph Frontend["Frontend (index.html)"]
        AssessTab["Patient Assessment Tab"]
        ChatTab["Chat Tab"]
        AdminTab["Admin Tab"]
    end

    subgraph FastAPI["FastAPI Service"]
        subgraph Routers["API Routers"]
            AssessRouter["/assess/*<br/>(GET patients, POST assess)"]
            ChatRouter["/chat/*<br/>(POST chat, GET/DELETE history)"]
            AdminRouter["/admin/*<br/>(refresh, stats, chunks, canonical)"]
        end

        subgraph Agents["LangGraph Workflows"]
            AssessWorkflow["assessment_workflow.py<br/>(4 nodes)"]
            ChatWorkflow["chat_workflow.py<br/>(12 nodes)"]
        end
    end

    subgraph SharedCore["Shared Core"]
        RAGPipeline["rag_pipeline.py<br/>(retrieve, _chat_rerank,<br/>_attach_canonicals)"]
        VectorStore["vector_store.py<br/>(ChromaDB wrapper)"]
        Embeddings["embeddings.py"]
        GeminiClient["gemini_client.py"]
        QueryBuilder2["query_builder.py<br/>(A+C+B tiers)"]
    end

    subgraph Storage["Storage"]
        ChromaDB["ChromaDB<br/>3 Collections:<br/>ng12_canonical<br/>ng12_search<br/>ng12_symptom"]
        SessionStore2["SessionStore<br/>(In-Memory)"]
        PatientDB2["patients.json<br/>(10 test patients)"]
    end

    subgraph External["External Services"]
        VertexAI["Vertex AI<br/>text-embedding-004"]
        Gemini["Gemini 2.0 Flash"]
    end

    Frontend --> FastAPI
    Routers --> Agents
    Agents --> SharedCore
    SharedCore --> Storage
    SharedCore --> External

    style SharedCore fill:#e3f2fd,stroke:#1976d2
    style ChromaDB fill:#fff3e0,stroke:#f57c00
```
