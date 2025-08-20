"""
Simple terminal visualization for hop journeys
"""
import time

def print_hop_banner():
    """Print a simple hop journey banner"""
    print("\n" + "=" * 80)
    print("🛤️  HOP JOURNEY VISUALIZATION")
    print("=" * 80)

def print_step(step_num: int, step_name: str, query: str, count: int, time_taken: float, details: str = ""):
    """Print a single step in the journey"""
    print(f"\n🔍 STEP {step_num}: {step_name}")
    print(f"├─ Query: {query[:60]}...")
    print(f"├─ Results: {count} chunks")
    print(f"├─ Time: {time_taken:.3f}s")
    if details:
        print(f"└─ Details: {details}")
    else:
        print("└─ ─ ─ ─ ─ ─ ─ ─")

def print_hop_summary(total_steps: int, total_time: float, final_count: int):
    """Print journey summary"""
    print(f"\n🎯 JOURNEY COMPLETE:")
    print(f"├─ Total Steps: {total_steps}")
    print(f"├─ Total Time: {total_time:.3f}s")
    print(f"└─ Final Results: {final_count} chunks")
    print("=" * 80 + "\n")

def print_simple_mermaid(query: str, steps: list):
    """Print a simple text-based flow diagram"""
    print("\n📊 FLOW DIAGRAM:")
    print("┌─────────────────────────────────────────────────┐")
    print(f"│ Query: {query[:40]:<40} │")
    print("└─────────────────────┬───────────────────────────┘")
    
    for i, step in enumerate(steps):
        if i == 0:
            print("                      │")
        print("                      ▼")
        step_name = step.get('name', f'Step {i+1}')[:25]
        step_count = step.get('count', 0)
        print(f"                ┌─────────────┐")
        print(f"                │ {step_name:<11} │")
        print(f"                │ {step_count:>3} chunks    │")
        print(f"                └─────────────┘")
    
    print()

# Global step tracking
_current_steps = []
_current_query = ""
_start_time = None

def start_simple_tracking(query: str):
    """Start simple step tracking"""
    global _current_steps, _current_query, _start_time
    _current_steps = []
    _current_query = query
    _start_time = time.time()
    print_hop_banner()

def log_simple_step(name: str, count: int, time_taken: float, details: str = "", metadata: dict = None):
    """Log a simple step"""
    global _current_steps
    step_num = len(_current_steps) + 1
    _current_steps.append({
        'name': name,
        'count': count,
        'time': time_taken,
        'details': details,
        'metadata': metadata or {}
    })
    print_step(step_num, name, _current_query, count, time_taken, details)
    
    # Show metadata if provided
    if metadata:
        for key, value in metadata.items():
            if isinstance(value, list) and value:
                print(f"      📋 {key}: {value[:3]}{'...' if len(value) > 3 else ''}")
            elif value:
                print(f"      📋 {key}: {value}")

def finish_simple_tracking():
    """Finish tracking and print summary"""
    global _current_steps, _start_time
    import time
    total_time = time.time() - _start_time if _start_time else 0
    final_count = _current_steps[-1]['count'] if _current_steps else 0
    print_hop_summary(len(_current_steps), total_time, final_count)
    print_simple_mermaid(_current_query, _current_steps)
