"""
Hop Visualization System
Creates visual diagrams and detailed logs of the 2-hop search journey
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class HopStep:
    """Represents a single step in the hop journey"""
    step_name: str
    step_type: str  # 'search', 'filter', 'signal_extraction', 'llm'
    input_query: str
    results_count: int
    timing: float
    metadata: Dict[str, Any]
    details: str

class HopVisualizer:
    """Creates visual representations of hop journeys"""
    
    def __init__(self):
        self.steps: List[HopStep] = []
        self.query = ""
        self.intent_type = ""
        self.start_time = None
        
    def start_journey(self, query: str, intent_type: str):
        """Start tracking a new hop journey"""
        self.query = query
        self.intent_type = intent_type
        self.start_time = datetime.now()
        self.steps = []
        logger.info(f"🛤️  Starting hop journey for: '{query[:50]}...'")
    
    def add_step(self, step_name: str, step_type: str, input_query: str, 
                 results_count: int, timing: float, metadata: Dict[str, Any] = None, 
                 details: str = ""):
        """Add a step to the journey"""
        step = HopStep(
            step_name=step_name,
            step_type=step_type,
            input_query=input_query,
            results_count=results_count,
            timing=timing,
            metadata=metadata or {},
            details=details
        )
        self.steps.append(step)
        
        # Log the step immediately
        self._log_step(step)
    
    def _log_step(self, step: HopStep):
        """Log a single step with rich details"""
        step_icon = {
            'search': '🔍',
            'filter': '🔧', 
            'signal_extraction': '🧠',
            'llm': '💬',
            'rerank': '📊'
        }.get(step.step_type, '⚡')
        
        logger.info(f"{step_icon} {step.step_name}")
        logger.info(f"   📋 Query: '{step.input_query[:60]}...'")
        logger.info(f"   📈 Results: {step.results_count} chunks")
        logger.info(f"   ⏱️  Time: {step.timing:.3f}s")
        
        if step.details:
            logger.info(f"   💡 Details: {step.details}")
        
        if step.metadata:
            for key, value in step.metadata.items():
                if isinstance(value, list) and len(value) > 3:
                    logger.info(f"   🔗 {key}: {value[:3]}... ({len(value)} total)")
                else:
                    logger.info(f"   🔗 {key}: {value}")
    
    def create_mermaid_diagram(self) -> str:
        """Create a Mermaid flowchart of the hop journey"""
        if not self.steps:
            return "No steps recorded"
        
        mermaid = ["graph TD"]
        mermaid.append(f'    START["🔍 Query: {self.query[:30]}..."]')
        mermaid.append(f'    INTENT["🎯 Intent: {self.intent_type}"]')
        mermaid.append("    START --> INTENT")
        
        prev_node = "INTENT"
        
        for i, step in enumerate(self.steps):
            node_id = f"STEP{i}"
            step_icon = {
                'search': '🔍',
                'filter': '🔧',
                'signal_extraction': '🧠', 
                'llm': '💬',
                'rerank': '📊'
            }.get(step.step_type, '⚡')
            
            # Create node with step info
            node_label = f'{step_icon} {step.step_name}\\n{step.results_count} chunks\\n{step.timing:.2f}s'
            mermaid.append(f'    {node_id}["{node_label}"]')
            mermaid.append(f"    {prev_node} --> {node_id}")
            
            # Add metadata as separate nodes if significant
            if step.metadata:
                for key, value in step.metadata.items():
                    if key in ['signals', 'filters', 'top_terms'] and value:
                        meta_node = f"META{i}_{key.upper()}"
                        if isinstance(value, list):
                            meta_text = f"📋 {key}:\\n{', '.join(map(str, value[:3]))}"
                            if len(value) > 3:
                                meta_text += f"\\n...+{len(value)-3} more"
                        else:
                            meta_text = f"📋 {key}: {value}"
                        
                        mermaid.append(f'    {meta_node}["{meta_text}"]')
                        mermaid.append(f"    {node_id} -.-> {meta_node}")
            
            prev_node = node_id
        
        # Add final result
        final_count = self.steps[-1].results_count if self.steps else 0
        total_time = sum(step.timing for step in self.steps)
        mermaid.append(f'    RESULT["✅ Final Result\\n{final_count} chunks\\n{total_time:.2f}s total"]')
        mermaid.append(f"    {prev_node} --> RESULT")
        
        return "\n".join(mermaid)
    
    def create_ascii_diagram(self) -> str:
        """Create an ASCII flowchart for terminal display"""
        if not self.steps:
            return "No journey recorded"
        
        ascii_diagram = []
        ascii_diagram.append("=" * 80)
        ascii_diagram.append(f"🛤️  HOP JOURNEY VISUALIZATION")
        ascii_diagram.append(f"Query: {self.query}")
        ascii_diagram.append(f"Intent: {self.intent_type}")
        ascii_diagram.append(f"Started: {self.start_time.strftime('%H:%M:%S') if self.start_time else 'Unknown'}")
        ascii_diagram.append("=" * 80)
        
        total_time = 0
        for i, step in enumerate(self.steps):
            total_time += step.timing
            
            # Step header
            step_icon = {
                'search': '🔍',
                'filter': '🔧',
                'signal_extraction': '🧠',
                'llm': '💬', 
                'rerank': '📊'
            }.get(step.step_type, '⚡')
            
            ascii_diagram.append(f"\n{step_icon} STEP {i+1}: {step.step_name}")
            ascii_diagram.append("┌" + "─" * 78 + "┐")
            ascii_diagram.append(f"│ Query: {step.input_query[:60]:<60} │")
            ascii_diagram.append(f"│ Results: {step.results_count:>5} chunks | Time: {step.timing:>8.3f}s | Total: {total_time:>8.3f}s │")
            
            if step.details:
                ascii_diagram.append(f"│ Details: {step.details[:65]:<65} │")
            
            # Add metadata in a structured way
            if step.metadata:
                ascii_diagram.append("├" + "─" * 78 + "┤")
                for key, value in step.metadata.items():
                    if isinstance(value, list):
                        if len(value) <= 3:
                            value_str = ", ".join(map(str, value))
                        else:
                            value_str = f"{', '.join(map(str, value[:3]))}... (+{len(value)-3} more)"
                    else:
                        value_str = str(value)
                    
                    line = f"│ {key}: {value_str}"
                    if len(line) > 78:
                        line = line[:75] + "..."
                    ascii_diagram.append(f"{line:<79}│")
            
            ascii_diagram.append("└" + "─" * 78 + "┘")
            
            # Add arrow to next step
            if i < len(self.steps) - 1:
                ascii_diagram.append("                                    ↓")
        
        # Summary
        ascii_diagram.append(f"\n🎯 JOURNEY SUMMARY:")
        ascii_diagram.append(f"   Total Steps: {len(self.steps)}")
        ascii_diagram.append(f"   Total Time: {total_time:.3f}s")
        ascii_diagram.append(f"   Final Results: {self.steps[-1].results_count if self.steps else 0} chunks")
        ascii_diagram.append("=" * 80)
        
        return "\n".join(ascii_diagram)
    
    def create_timeline_view(self) -> str:
        """Create a timeline view of the journey"""
        if not self.steps:
            return "No timeline available"
        
        timeline = []
        timeline.append("⏰ HOP TIMELINE")
        timeline.append("═" * 50)
        
        cumulative_time = 0
        for i, step in enumerate(self.steps):
            cumulative_time += step.timing
            
            # Timeline entry
            timeline.append(f"{cumulative_time:>6.2f}s │ {step.step_name}")
            timeline.append(f"        │ └─ {step.results_count} results ({step.timing:.3f}s)")
            
            if step.metadata and 'top_signal' in step.metadata:
                timeline.append(f"        │    Signal: {step.metadata['top_signal']}")
            
            if i < len(self.steps) - 1:
                timeline.append("        │")
        
        return "\n".join(timeline)
    
    def print_journey(self, style: str = "ascii"):
        """Print the complete journey visualization"""
        if style == "mermaid":
            print("\n🧩 MERMAID DIAGRAM:")
            print(self.create_mermaid_diagram())
        elif style == "timeline": 
            print("\n" + self.create_timeline_view())
        else:  # ascii (default)
            print("\n" + self.create_ascii_diagram())

# Global visualizer instance
visualizer = HopVisualizer()

def start_visualization(query: str, intent_type: str):
    """Start hop visualization"""
    visualizer.start_journey(query, intent_type)

def log_hop_step(step_name: str, step_type: str, input_query: str,
                 results_count: int, timing: float, metadata: Dict[str, Any] = None,
                 details: str = ""):
    """Log a hop step with visualization"""
    visualizer.add_step(step_name, step_type, input_query, results_count, timing, metadata, details)

def print_hop_journey(style: str = "ascii"):
    """Print the complete hop journey"""
    visualizer.print_journey(style)

def get_mermaid_diagram() -> str:
    """Get mermaid diagram as string"""
    return visualizer.create_mermaid_diagram()
