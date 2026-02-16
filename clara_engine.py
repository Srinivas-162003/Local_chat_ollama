"""
CLaRa (Contextualized Language Retrieval and Answering) Engine

Advanced RAG system with:
- Iterative retrieval and query refinement
- Multi-hop reasoning across documents
- Ambiguity detection and resolution
- Evidence tracking and attribution
- Context-aware refinement loops
"""

import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
try:
    from langchain_ollama import OllamaLLM
except ImportError:
    from langchain_community.llms import Ollama as OllamaLLM
from langchain_core.prompts import PromptTemplate
from processor import get_retriever

logger = logging.getLogger(__name__)

_llm = None
_clara_engine = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = OllamaLLM(model="llama3.2:latest")
    return _llm


@dataclass
class RetrievedEvidence:
    """Evidence from a single retrieval step"""
    content: str
    source: str
    relevance_score: float
    retrieval_step: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningStep:
    """A single step in multi-hop reasoning"""
    step_number: int
    query: str
    evidence: List[RetrievedEvidence]
    intermediate_answer: str
    confidence: float
    identified_gaps: List[str] = field(default_factory=list)


@dataclass
class CLaRaResponse:
    """Complete CLaRa response with full reasoning trail"""
    final_answer: str
    reasoning_steps: List[ReasoningStep]
    total_iterations: int
    clarifications_needed: List[str]
    evidence_map: Dict[str, List[str]]  # claim -> source mappings
    confidence_score: float


class QueryAnalyzer:
    """Analyzes queries for complexity, ambiguity, and multi-hop requirements"""
    
    ANALYSIS_PROMPT = """Analyze this question and determine:
1. Is it ambiguous or unclear? (yes/no)
2. Does it require multi-hop reasoning across multiple documents? (yes/no)
3. What are the key entities or concepts to search for?
4. Are there any implicit assumptions that need clarification?

Question: {question}

Provide analysis in this format:
AMBIGUOUS: [yes/no]
MULTI_HOP: [yes/no]
KEY_CONCEPTS: [comma-separated list]
ASSUMPTIONS: [list any assumptions or provide 'none']
SUGGESTED_CLARIFICATIONS: [questions to ask user, or 'none']
"""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            template=self.ANALYSIS_PROMPT,
            input_variables=["question"]
        )
    
    def analyze(self, question: str) -> Dict[str, Any]:
        """Analyze query complexity and ambiguity"""
        try:
            analysis_text = self.llm.invoke(self.prompt.format(question=question))
            
            # Parse the structured response
            result = {
                "is_ambiguous": "yes" in analysis_text.lower().split("ambiguous:")[1].split("\n")[0].lower(),
                "requires_multi_hop": "yes" in analysis_text.lower().split("multi_hop:")[1].split("\n")[0].lower(),
                "key_concepts": [],
                "assumptions": [],
                "clarifications": []
            }
            
            # Extract key concepts
            if "key_concepts:" in analysis_text.lower():
                concepts_line = analysis_text.split("KEY_CONCEPTS:")[1].split("\n")[0]
                result["key_concepts"] = [c.strip() for c in concepts_line.split(",") if c.strip()]
            
            # Extract clarifications
            if "suggested_clarifications:" in analysis_text.lower():
                clarif_line = analysis_text.split("SUGGESTED_CLARIFICATIONS:")[1].split("\n")[0]
                if "none" not in clarif_line.lower():
                    result["clarifications"] = [clarif_line.strip()]
            
            return result
            
        except Exception as e:
            logger.error(f"Query analysis error: {e}")
            return {
                "is_ambiguous": False,
                "requires_multi_hop": False,
                "key_concepts": [question],
                "assumptions": [],
                "clarifications": []
            }


class IterativeRetriever:
    """Performs multi-pass retrieval with query refinement"""
    
    REFINEMENT_PROMPT = """Based on the question and what we've learned so far, generate an improved search query.

Original Question: {original_question}

Previous Findings: {previous_findings}

Identified Gaps: {gaps}

Generate a refined search query that will help fill the gaps in our understanding.
Only output the refined query, nothing else."""
    
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.refinement_prompt = PromptTemplate(
            template=self.REFINEMENT_PROMPT,
            input_variables=["original_question", "previous_findings", "gaps"]
        )
    
    def retrieve_with_refinement(
        self, 
        original_query: str, 
        max_iterations: int = 3,
        previous_findings: str = ""
    ) -> List[RetrievedEvidence]:
        """Iteratively retrieve and refine"""
        all_evidence = []
        current_query = original_query
        
        for iteration in range(max_iterations):
            logger.info(f"CLaRa Retrieval iteration {iteration + 1}: {current_query}")
            
            # Retrieve documents
            docs = self.retriever.invoke(current_query)
            
            # Convert to evidence objects
            for idx, doc in enumerate(docs):
                evidence = RetrievedEvidence(
                    content=doc.page_content,
                    source=doc.metadata.get("source_file", "unknown"),
                    relevance_score=1.0 / (idx + 1),  # Simple relevance scoring
                    retrieval_step=iteration + 1,
                    metadata=doc.metadata
                )
                all_evidence.append(evidence)
            
            # If this is not the last iteration, refine the query
            if iteration < max_iterations - 1:
                # Check if we have enough diverse information
                unique_sources = set(e.source for e in all_evidence)
                if len(unique_sources) >= 3 and len(all_evidence) >= 10:
                    logger.info("Sufficient evidence gathered, stopping early")
                    break
                
                # Refine query for next iteration
                findings_summary = self._summarize_findings(all_evidence)
                gaps = self._identify_gaps(original_query, findings_summary)
                
                if not gaps:
                    logger.info("No gaps identified, stopping iteration")
                    break
                
                current_query = self._refine_query(
                    original_query, 
                    findings_summary, 
                    gaps
                )
        
        return all_evidence
    
    def _summarize_findings(self, evidence: List[RetrievedEvidence]) -> str:
        """Summarize what we've found so far"""
        if not evidence:
            return "No findings yet"
        
        # Take top pieces of evidence
        top_evidence = sorted(evidence, key=lambda e: e.relevance_score, reverse=True)[:5]
        summary = "\n".join([f"- {e.content[:150]}..." for e in top_evidence])
        return summary
    
    def _identify_gaps(self, query: str, findings: str) -> List[str]:
        """Identify information gaps"""
        # Simple gap detection - in production, use LLM
        if len(findings) < 100:
            return ["Need more detailed information"]
        return []
    
    def _refine_query(self, original: str, findings: str, gaps: List[str]) -> str:
        """Generate refined query"""
        try:
            refined = self.llm.invoke(
                self.refinement_prompt.format(
                    original_question=original,
                    previous_findings=findings[:500],
                    gaps=", ".join(gaps) if gaps else "Need more context"
                )
            )
            return refined.strip()
        except Exception as e:
            logger.error(f"Query refinement error: {e}")
            return original


class MultiHopReasoner:
    """Performs step-by-step reasoning across multiple documents"""
    
    REASONING_PROMPT = """You are performing step-by-step reasoning to answer a complex question.

Original Question: {question}

Step {step_number} Query: {current_query}

Available Evidence:
{evidence}

Previous Steps:
{previous_steps}

Based on the evidence above:
1. Answer what you can for this step
2. Identify what information is still missing
3. Rate your confidence (0-1)

Format your response as:
ANSWER: [your answer for this step]
MISSING: [what's still needed, or 'none']
CONFIDENCE: [0.0-1.0]
NEXT_QUERY: [suggested next search query, or 'none' if complete]
"""
    
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        self.prompt = PromptTemplate(
            template=self.REASONING_PROMPT,
            input_variables=["question", "step_number", "current_query", "evidence", "previous_steps"]
        )
    
    def reason_multi_hop(
        self, 
        question: str, 
        initial_evidence: List[RetrievedEvidence],
        max_hops: int = 3
    ) -> List[ReasoningStep]:
        """Perform multi-hop reasoning"""
        reasoning_steps = []
        current_query = question
        
        for hop in range(max_hops):
            logger.info(f"Reasoning hop {hop + 1}/{max_hops}")
            
            # Use initial evidence for first hop, retrieve new for subsequent
            if hop == 0:
                evidence = initial_evidence
            else:
                docs = self.retriever.invoke(current_query)
                evidence = [
                    RetrievedEvidence(
                        content=doc.page_content,
                        source=doc.metadata.get("source_file", "unknown"),
                        relevance_score=1.0,
                        retrieval_step=hop + 1,
                        metadata=doc.metadata
                    )
                    for doc in docs
                ]
            
            # Format evidence for reasoning
            evidence_text = "\n\n".join([
                f"[{e.source}] {e.content[:300]}"
                for e in evidence[:5]
            ])
            
            # Format previous steps
            previous_steps_text = "\n".join([
                f"Step {s.step_number}: {s.intermediate_answer}"
                for s in reasoning_steps
            ]) if reasoning_steps else "This is the first step"
            
            # Perform reasoning
            try:
                reasoning_output = self.llm.invoke(
                    self.prompt.format(
                        question=question,
                        step_number=hop + 1,
                        current_query=current_query,
                        evidence=evidence_text,
                        previous_steps=previous_steps_text
                    )
                )
                
                # Parse reasoning output
                answer, confidence, next_query, gaps = self._parse_reasoning(reasoning_output)
                
                step = ReasoningStep(
                    step_number=hop + 1,
                    query=current_query,
                    evidence=evidence,
                    intermediate_answer=answer,
                    confidence=confidence,
                    identified_gaps=gaps
                )
                reasoning_steps.append(step)
                
                # Check if we should continue
                if next_query.lower() == "none" or confidence > 0.9:
                    logger.info(f"Reasoning complete at hop {hop + 1}")
                    break
                
                current_query = next_query
                
            except Exception as e:
                logger.error(f"Reasoning error at hop {hop + 1}: {e}")
                break
        
        return reasoning_steps
    
    def _parse_reasoning(self, output: str) -> Tuple[str, float, str, List[str]]:
        """Parse structured reasoning output"""
        try:
            answer = output.split("ANSWER:")[1].split("MISSING:")[0].strip()
            confidence_str = output.split("CONFIDENCE:")[1].split("\n")[0].strip()
            confidence = float(confidence_str)
            next_query = output.split("NEXT_QUERY:")[1].strip() if "NEXT_QUERY:" in output else "none"
            
            missing_section = output.split("MISSING:")[1].split("CONFIDENCE:")[0].strip()
            gaps = [missing_section] if missing_section.lower() != "none" else []
            
            return answer, confidence, next_query, gaps
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return output[:200], 0.5, "none", []


class EvidenceTracker:
    """Tracks evidence-to-claim mappings for attribution"""
    
    def build_evidence_map(self, reasoning_steps: List[ReasoningStep]) -> Dict[str, List[str]]:
        """Build mapping of claims to their source evidence"""
        evidence_map = {}
        
        for step in reasoning_steps:
            claim = step.intermediate_answer[:100]  # Use first 100 chars as key
            sources = list(set([e.source for e in step.evidence]))
            evidence_map[f"Step {step.step_number}"] = sources
        
        return evidence_map


class CLaRaEngine:
    """Main CLaRa engine coordinating all components"""
    
    SYNTHESIS_PROMPT = """Synthesize a final comprehensive answer from multi-hop reasoning steps.

Original Question: {question}

Reasoning Steps:
{reasoning_steps}

Provide a complete, well-structured answer that:
1. Directly answers the question
2. Integrates information from all reasoning steps
3. Maintains factual accuracy
4. Cites sources when making specific claims

Final Answer:"""
    
    def __init__(self):
        self.llm = _get_llm()
        self.retriever = get_retriever()
        self.query_analyzer = QueryAnalyzer(self.llm)
        self.iterative_retriever = IterativeRetriever(self.retriever, self.llm)
        self.multi_hop_reasoner = MultiHopReasoner(self.llm, self.retriever)
        self.evidence_tracker = EvidenceTracker()
        self.synthesis_prompt = PromptTemplate(
            template=self.SYNTHESIS_PROMPT,
            input_variables=["question", "reasoning_steps"]
        )
    
    def answer(
        self, 
        question: str, 
        max_iterations: int = 3,
        max_hops: int = 3,
        enable_clarification: bool = True
    ) -> CLaRaResponse:
        """
        Main CLaRa answering pipeline
        
        Args:
            question: The user's question
            max_iterations: Maximum retrieval iterations
            max_hops: Maximum reasoning hops
            enable_clarification: Whether to suggest clarifications
        """
        logger.info(f"CLaRa processing: {question}")
        
        # Step 1: Analyze query
        analysis = self.query_analyzer.analyze(question)
        logger.info(f"Query analysis: {analysis}")
        
        clarifications = analysis["clarifications"] if enable_clarification else []
        
        # Step 2: Iterative retrieval with refinement
        evidence = self.iterative_retriever.retrieve_with_refinement(
            question, 
            max_iterations=max_iterations
        )
        logger.info(f"Retrieved {len(evidence)} pieces of evidence across iterations")
        
        # Step 3: Multi-hop reasoning (if needed)
        if analysis["requires_multi_hop"] or len(evidence) > 10:
            reasoning_steps = self.multi_hop_reasoner.reason_multi_hop(
                question, 
                evidence, 
                max_hops=max_hops
            )
        else:
            # Simple single-step reasoning
            reasoning_steps = [
                ReasoningStep(
                    step_number=1,
                    query=question,
                    evidence=evidence,
                    intermediate_answer=self._simple_answer(question, evidence),
                    confidence=0.8,
                    identified_gaps=[]
                )
            ]
        
        # Step 4: Build evidence map
        evidence_map = self.evidence_tracker.build_evidence_map(reasoning_steps)
        
        # Step 5: Synthesize final answer
        final_answer = self._synthesize_answer(question, reasoning_steps)
        
        # Step 6: Calculate overall confidence
        avg_confidence = sum(s.confidence for s in reasoning_steps) / len(reasoning_steps)
        
        return CLaRaResponse(
            final_answer=final_answer,
            reasoning_steps=reasoning_steps,
            total_iterations=len(set(e.retrieval_step for e in evidence)),
            clarifications_needed=clarifications,
            evidence_map=evidence_map,
            confidence_score=avg_confidence
        )
    
    def _simple_answer(self, question: str, evidence: List[RetrievedEvidence]) -> str:
        """Generate simple answer for non-multi-hop questions"""
        context = "\n\n".join([f"[{e.source}] {e.content}" for e in evidence[:5]])
        
        prompt = f"""Answer the question based on the provided context.

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            return self.llm.invoke(prompt)
        except Exception as e:
            logger.error(f"Simple answer generation error: {e}")
            return "Unable to generate answer from available evidence."
    
    def _synthesize_answer(self, question: str, reasoning_steps: List[ReasoningStep]) -> str:
        """Synthesize final answer from reasoning steps"""
        steps_text = "\n\n".join([
            f"Step {s.step_number} (confidence: {s.confidence:.2f}):\n{s.intermediate_answer}"
            for s in reasoning_steps
        ])
        
        try:
            final = self.llm.invoke(
                self.synthesis_prompt.format(
                    question=question,
                    reasoning_steps=steps_text
                )
            )
            return final
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            # Fallback: return last reasoning step
            return reasoning_steps[-1].intermediate_answer if reasoning_steps else ""


def get_clara_engine() -> CLaRaEngine:
    global _clara_engine
    if _clara_engine is None:
        _clara_engine = CLaRaEngine()
    return _clara_engine


def answer_with_clara(
    question: str, 
    max_iterations: int = 3,
    max_hops: int = 3,
    detailed_response: bool = False
) -> str | Dict[str, Any]:
    """
    Answer using CLaRa engine
    
    Args:
        question: User's question
        max_iterations: Max retrieval iterations
        max_hops: Max reasoning hops
        detailed_response: If True, return full CLaRaResponse details
    
    Returns:
        String answer or detailed response dict
    """
    try:
        response = get_clara_engine().answer(
            question, 
            max_iterations=max_iterations,
            max_hops=max_hops
        )
        
        if detailed_response:
            return {
                "answer": response.final_answer,
                "reasoning_steps": [
                    {
                        "step": s.step_number,
                        "query": s.query,
                        "answer": s.intermediate_answer,
                        "confidence": s.confidence,
                        "sources": list(set(e.source for e in s.evidence))
                    }
                    for s in response.reasoning_steps
                ],
                "total_iterations": response.total_iterations,
                "confidence": response.confidence_score,
                "clarifications": response.clarifications_needed,
                "evidence_map": response.evidence_map
            }
        else:
            # Return just the answer for simple usage
            return response.final_answer
            
    except Exception as e:
        logger.error(f"CLaRa error: {e}")
        raise
