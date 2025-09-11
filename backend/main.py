from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import time
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Enhanced Arize AX observability setup
try:
    from arize.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.litellm import LiteLLMInstrumentor
    from openinference.instrumentation import using_prompt_template, using_session
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    import uuid
    _TRACING = True
except Exception as e:
    print(f"Warning: Observability imports failed: {e}")
    def using_prompt_template(**kwargs):  # type: ignore
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    def using_session(**kwargs):  # type: ignore
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    _TRACING = False

# LangGraph + LangChain
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


class TripRequest(BaseModel):
    destination: str
    duration: str
    budget: Optional[str] = None
    interests: Optional[str] = None
    travel_style: Optional[str] = None


class TripResponse(BaseModel):
    result: str
    tool_calls: List[Dict[str, Any]] = []


def _init_llm():
    # Simple, test-friendly LLM init
    class _Fake:
        def __init__(self):
            pass
        def bind_tools(self, tools):
            return self
        def invoke(self, messages):
            class _Msg:
                content = "Test itinerary"
                tool_calls: List[Dict[str, Any]] = []
            return _Msg()

    if os.getenv("TEST_MODE"):
        return _Fake()
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=1500)
    elif os.getenv("OPENROUTER_API_KEY"):
        # Use OpenRouter via OpenAI-compatible client
        return ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            temperature=0.7,
        )
    else:
        # Require a key unless running tests
        raise ValueError("Please set OPENAI_API_KEY or OPENROUTER_API_KEY in your .env")


llm = _init_llm()


# Minimal tools (deterministic for tutorials)
@tool
def essential_info(destination: str) -> str:
    """Return essential destination info like weather, sights, and etiquette."""
    # Enhanced mock data with actual structure
    return f"""Essential Information for {destination}:
    - Climate: Tropical/temperate with seasonal variations
    - Best time to visit: Spring and fall months
    - Top attractions: Historical sites, natural landmarks, cultural centers
    - Local customs: Respectful dress at religious sites, tipping culture varies
    - Language: Local language with English widely spoken in tourist areas
    - Currency: Local currency, credit cards accepted in most establishments
    - Safety: Generally safe for tourists, standard precautions advised"""


@tool
def budget_basics(destination: str, duration: str) -> str:
    """Return high-level budget categories for a given destination and duration."""
    return f"""Budget breakdown for {destination} ({duration}):
    - Accommodation: $50-200/night depending on style
    - Meals: $30-80/day (street food to restaurants)
    - Local transport: $10-30/day (public transit to taxis)
    - Activities/attractions: $20-60/day
    - Shopping/extras: $20-50/day
    Total estimated daily budget: $130-420 depending on travel style"""


@tool
def local_flavor(destination: str, interests: Optional[str] = None) -> str:
    """Suggest authentic local experiences matching optional interests."""
    interest_str = f" focusing on {interests}" if interests else ""
    return f"""Authentic local experiences in {destination}{interest_str}:
    - Morning markets with local vendors and fresh produce
    - Traditional cooking classes with local families
    - Neighborhood walking tours off the beaten path
    - Local artisan workshops and craft demonstrations
    - Community cultural performances and festivals
    - Hidden cafes and restaurants favored by locals
    - Sacred sites and temples with cultural significance"""


@tool
def day_plan(destination: str, day: int) -> str:
    """Return a simple day plan outline for a specific day number."""
    return f"Day {day} in {destination}: breakfast, highlight visit, lunch, afternoon walk, dinner."


# Additional simple tools per agent (to mirror original multi-tool behavior)
@tool
def weather_brief(destination: str) -> str:
    """Return a brief weather summary for planning purposes."""
    return f"""Weather overview for {destination}:
    - Current season: Varies by hemisphere and elevation
    - Temperature range: 20-30°C (68-86°F) typical
    - Rainfall: Seasonal patterns, pack rain gear if visiting in wet season
    - Humidity: Moderate to high in tropical areas
    - Pack: Layers, sun protection, comfortable walking shoes"""


@tool
def visa_brief(destination: str) -> str:
    """Return a brief visa guidance placeholder for tutorial purposes."""
    return f"Visa guidance for {destination}: check your nationality's embassy site."


@tool
def attraction_prices(destination: str, attractions: Optional[List[str]] = None) -> str:
    """Return rough placeholder prices for attractions."""
    items = attractions or ["Museum", "Historic Site", "Viewpoint"]
    priced = "\n    - ".join(f"{a}: $10-40 per person" for a in items)
    return f"""Attraction pricing in {destination}:
    - {priced}
    - Multi-day passes: Often 20-30% savings
    - Student/senior discounts: Usually 25-50% off
    - Free days: Many museums offer free entry certain days/hours
    - Booking online: Can save 10-15% vs gate prices"""


@tool
def local_customs(destination: str) -> str:
    """Return simple etiquette reminders for the destination."""
    return f"Customs in {destination}: be polite, modest dress in sacred places, learn greetings."


@tool
def hidden_gems(destination: str) -> str:
    """Return a few off-the-beaten-path ideas."""
    return f"""Hidden gems in {destination}:
    - Secret sunrise viewpoint known mainly to locals
    - Family-run restaurant with no sign (ask locals for directions)
    - Abandoned temple/building with incredible architecture
    - Local swimming hole or beach away from tourist crowds  
    - Artisan quarter where craftsmen still work traditionally
    - Night market that only operates certain days
    - Community garden or park perfect for picnics"""


@tool
def travel_time(from_location: str, to_location: str, mode: str = "public") -> str:
    """Return an approximate travel time placeholder."""
    return f"Travel from {from_location} to {to_location} by {mode}: ~20-60 minutes."


@tool
def packing_list(destination: str, duration: str, activities: Optional[List[str]] = None) -> str:
    """Return a generic packing list summary."""
    acts = ", ".join(activities or ["walking", "sightseeing"]) 
    return f"Packing for {destination} ({duration}): comfortable shoes, layers, adapter; for {acts}."


class TripState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    trip_request: Dict[str, Any]
    research: Optional[str]
    budget: Optional[str]
    local: Optional[str]
    final: Optional[str]
    tool_calls: Annotated[List[Dict[str, Any]], operator.add]


def research_agent(state: TripState) -> TripState:
    if tracer and _TRACING:
        with tracer.start_as_current_span("research_agent") as span:
            req = state["trip_request"]
            destination = req["destination"]
            
            span.set_attributes({
                "agent.name": "research",
                "agent.destination": destination,
                "agent.tools": "essential_info,weather_brief,visa_brief"
            })
            
            prompt_t = (
                "You are a research assistant.\n"
                "Gather essential information about {destination}.\n"
                "Use tools to get weather, visa, and essential info, then summarize."
            )
            vars_ = {"destination": destination}
            
            messages = [SystemMessage(content=prompt_t.format(**vars_))]
            tools = [essential_info, weather_brief, visa_brief]
            agent = llm.bind_tools(tools)
            
            calls: List[Dict[str, Any]] = []
            tool_results = []
            
            with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
                res = agent.invoke(messages)
            
            # Collect tool calls and execute them
            if getattr(res, "tool_calls", None):
                span.set_attribute("agent.tool_calls_count", len(res.tool_calls))
                for c in res.tool_calls:
                    calls.append({"agent": "research", "tool": c["name"], "args": c.get("args", {})})
                
                tool_node = ToolNode(tools)
                tr = tool_node.invoke({"messages": [res]})
                tool_results = tr["messages"]
                
                # Add tool results to conversation and ask LLM to synthesize
                messages.append(res)
                messages.extend(tool_results)
                messages.append(SystemMessage(content="Based on the above information, provide a comprehensive summary for the traveler."))
                
                # Get final synthesis from LLM
                final_res = llm.invoke(messages)
                out = final_res.content
            else:
                span.set_attribute("agent.tool_calls_count", 0)
                out = res.content

            span.set_attributes({
                "agent.output_length": len(out),
                "agent.status": "completed"
            })
            
            return {"messages": [SystemMessage(content=out)], "research": out, "tool_calls": calls}
    else:
        # Fallback without tracing
        req = state["trip_request"]
        destination = req["destination"]
        prompt_t = (
            "You are a research assistant.\n"
            "Gather essential information about {destination}.\n"
            "Use tools to get weather, visa, and essential info, then summarize."
        )
        vars_ = {"destination": destination}
        
        messages = [SystemMessage(content=prompt_t.format(**vars_))]
        tools = [essential_info, weather_brief, visa_brief]
        agent = llm.bind_tools(tools)
        
        calls: List[Dict[str, Any]] = []
        tool_results = []
        
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = agent.invoke(messages)
        
        if getattr(res, "tool_calls", None):
            for c in res.tool_calls:
                calls.append({"agent": "research", "tool": c["name"], "args": c.get("args", {})})
            
            tool_node = ToolNode(tools)
            tr = tool_node.invoke({"messages": [res]})
            tool_results = tr["messages"]
            
            messages.append(res)
            messages.extend(tool_results)
            messages.append(SystemMessage(content="Based on the above information, provide a comprehensive summary for the traveler."))
            
            final_res = llm.invoke(messages)
            out = final_res.content
        else:
            out = res.content

        return {"messages": [SystemMessage(content=out)], "research": out, "tool_calls": calls}


def budget_agent(state: TripState) -> TripState:
    if tracer and _TRACING:
        with tracer.start_as_current_span("budget_agent") as span:
            req = state["trip_request"]
            destination, duration = req["destination"], req["duration"]
            budget = req.get("budget", "moderate")
            
            span.set_attributes({
                "agent.name": "budget",
                "agent.destination": destination,
                "agent.duration": duration,
                "agent.budget": budget,
                "agent.tools": "budget_basics,attraction_prices"
            })
            
            prompt_t = (
                "You are a budget analyst.\n"
                "Analyze costs for {destination} over {duration} with budget: {budget}.\n"
                "Use tools to get pricing information, then provide a detailed breakdown."
            )
            vars_ = {"destination": destination, "duration": duration, "budget": budget}
            
            messages = [SystemMessage(content=prompt_t.format(**vars_))]
            tools = [budget_basics, attraction_prices]
            agent = llm.bind_tools(tools)
            
            calls: List[Dict[str, Any]] = []
            
            with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
                res = agent.invoke(messages)
            
            if getattr(res, "tool_calls", None):
                span.set_attribute("agent.tool_calls_count", len(res.tool_calls))
                for c in res.tool_calls:
                    calls.append({"agent": "budget", "tool": c["name"], "args": c.get("args", {})})
                
                tool_node = ToolNode(tools)
                tr = tool_node.invoke({"messages": [res]})
                
                # Add tool results and ask for synthesis
                messages.append(res)
                messages.extend(tr["messages"])
                messages.append(SystemMessage(content=f"Create a detailed budget breakdown for {duration} in {destination} with a {budget} budget."))
                
                final_res = llm.invoke(messages)
                out = final_res.content
            else:
                span.set_attribute("agent.tool_calls_count", 0)
                out = res.content

            span.set_attributes({
                "agent.output_length": len(out),
                "agent.status": "completed"
            })
            
            return {"messages": [SystemMessage(content=out)], "budget": out, "tool_calls": calls}
    else:
        # Fallback without tracing
        req = state["trip_request"]
        destination, duration = req["destination"], req["duration"]
        budget = req.get("budget", "moderate")
        prompt_t = (
            "You are a budget analyst.\n"
            "Analyze costs for {destination} over {duration} with budget: {budget}.\n"
            "Use tools to get pricing information, then provide a detailed breakdown."
        )
        vars_ = {"destination": destination, "duration": duration, "budget": budget}
        
        messages = [SystemMessage(content=prompt_t.format(**vars_))]
        tools = [budget_basics, attraction_prices]
        agent = llm.bind_tools(tools)
        
        calls: List[Dict[str, Any]] = []
        
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = agent.invoke(messages)
        
        if getattr(res, "tool_calls", None):
            for c in res.tool_calls:
                calls.append({"agent": "budget", "tool": c["name"], "args": c.get("args", {})})
            
            tool_node = ToolNode(tools)
            tr = tool_node.invoke({"messages": [res]})
            
            messages.append(res)
            messages.extend(tr["messages"])
            messages.append(SystemMessage(content=f"Create a detailed budget breakdown for {duration} in {destination} with a {budget} budget."))
            
            final_res = llm.invoke(messages)
            out = final_res.content
        else:
            out = res.content

        return {"messages": [SystemMessage(content=out)], "budget": out, "tool_calls": calls}


def local_agent(state: TripState) -> TripState:
    if tracer and _TRACING:
        with tracer.start_as_current_span("local_agent") as span:
            req = state["trip_request"]
            destination = req["destination"]
            interests = req.get("interests", "local culture")
            travel_style = req.get("travel_style", "standard")
            
            span.set_attributes({
                "agent.name": "local",
                "agent.destination": destination,
                "agent.interests": interests,
                "agent.travel_style": travel_style,
                "agent.tools": "local_flavor,local_customs,hidden_gems"
            })
            
            prompt_t = (
                "You are a local guide.\n"
                "Find authentic experiences in {destination} for someone interested in: {interests}.\n"
                "Travel style: {travel_style}. Use tools to gather local insights."
            )
            vars_ = {"destination": destination, "interests": interests, "travel_style": travel_style}
            
            messages = [SystemMessage(content=prompt_t.format(**vars_))]
            tools = [local_flavor, local_customs, hidden_gems]
            agent = llm.bind_tools(tools)
            
            calls: List[Dict[str, Any]] = []
            
            with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
                res = agent.invoke(messages)
            
            if getattr(res, "tool_calls", None):
                span.set_attribute("agent.tool_calls_count", len(res.tool_calls))
                for c in res.tool_calls:
                    calls.append({"agent": "local", "tool": c["name"], "args": c.get("args", {})})
                
                tool_node = ToolNode(tools)
                tr = tool_node.invoke({"messages": [res]})
                
                # Add tool results and ask for synthesis
                messages.append(res)
                messages.extend(tr["messages"])
                messages.append(SystemMessage(content=f"Create a curated list of authentic experiences for someone interested in {interests} with a {travel_style} approach."))
                
                final_res = llm.invoke(messages)
                out = final_res.content
            else:
                span.set_attribute("agent.tool_calls_count", 0)
                out = res.content

            span.set_attributes({
                "agent.output_length": len(out),
                "agent.status": "completed"
            })
            
            return {"messages": [SystemMessage(content=out)], "local": out, "tool_calls": calls}
    else:
        # Fallback without tracing
        req = state["trip_request"]
        destination = req["destination"]
        interests = req.get("interests", "local culture")
        travel_style = req.get("travel_style", "standard")
        prompt_t = (
            "You are a local guide.\n"
            "Find authentic experiences in {destination} for someone interested in: {interests}.\n"
            "Travel style: {travel_style}. Use tools to gather local insights."
        )
        vars_ = {"destination": destination, "interests": interests, "travel_style": travel_style}
        
        messages = [SystemMessage(content=prompt_t.format(**vars_))]
        tools = [local_flavor, local_customs, hidden_gems]
        agent = llm.bind_tools(tools)
        
        calls: List[Dict[str, Any]] = []
        
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = agent.invoke(messages)
        
        if getattr(res, "tool_calls", None):
            for c in res.tool_calls:
                calls.append({"agent": "local", "tool": c["name"], "args": c.get("args", {})})
            
            tool_node = ToolNode(tools)
            tr = tool_node.invoke({"messages": [res]})
            
            messages.append(res)
            messages.extend(tr["messages"])
            messages.append(SystemMessage(content=f"Create a curated list of authentic experiences for someone interested in {interests} with a {travel_style} approach."))
            
            final_res = llm.invoke(messages)
            out = final_res.content
        else:
            out = res.content

        return {"messages": [SystemMessage(content=out)], "local": out, "tool_calls": calls}


def itinerary_agent(state: TripState) -> TripState:
    if tracer and _TRACING:
        with tracer.start_as_current_span("itinerary_agent") as span:
            req = state["trip_request"]
            destination = req["destination"]
            duration = req["duration"]
            travel_style = req.get("travel_style", "standard")
            
            # Get input lengths for metrics
            research_input = state.get("research") or ""
            budget_input = state.get("budget") or ""
            local_input = state.get("local") or ""
            
            span.set_attributes({
                "agent.name": "itinerary",
                "agent.destination": destination,
                "agent.duration": duration,
                "agent.travel_style": travel_style,
                "agent.research_input_length": len(research_input),
                "agent.budget_input_length": len(budget_input),
                "agent.local_input_length": len(local_input),
                "agent.synthesis": True
            })
            
            prompt_t = (
                "Create a {duration} itinerary for {destination} ({travel_style}).\n\n"
                "Inputs:\nResearch: {research}\nBudget: {budget}\nLocal: {local}\n"
            )
            vars_ = {
                "duration": duration,
                "destination": destination,
                "travel_style": travel_style,
                "research": research_input[:400],
                "budget": budget_input[:400],
                "local": local_input[:400],
            }
            
            with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
                res = llm.invoke([SystemMessage(content=prompt_t.format(**vars_))])
            
            span.set_attributes({
                "agent.output_length": len(res.content),
                "agent.status": "completed"
            })
            
            return {"messages": [SystemMessage(content=res.content)], "final": res.content}
    else:
        # Fallback without tracing
        req = state["trip_request"]
        destination = req["destination"]
        duration = req["duration"]
        travel_style = req.get("travel_style", "standard")
        prompt_t = (
            "Create a {duration} itinerary for {destination} ({travel_style}).\n\n"
            "Inputs:\nResearch: {research}\nBudget: {budget}\nLocal: {local}\n"
        )
        vars_ = {
            "duration": duration,
            "destination": destination,
            "travel_style": travel_style,
            "research": (state.get("research") or "")[:400],
            "budget": (state.get("budget") or "")[:400],
            "local": (state.get("local") or "")[:400],
        }
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = llm.invoke([SystemMessage(content=prompt_t.format(**vars_))])
        return {"messages": [SystemMessage(content=res.content)], "final": res.content}


def build_graph():
    g = StateGraph(TripState)
    g.add_node("research", research_agent)
    g.add_node("budget", budget_agent)
    g.add_node("local", local_agent)
    g.add_node("itinerary", itinerary_agent)

    # Run research, budget, and local agents in parallel
    g.add_edge(START, "research")
    g.add_edge(START, "budget")
    g.add_edge(START, "local")
    
    # All three agents feed into the itinerary agent
    g.add_edge("research", "itinerary")
    g.add_edge("budget", "itinerary")
    g.add_edge("local", "itinerary")
    
    g.add_edge("itinerary", END)

    # Compile without checkpointer to avoid state persistence issues
    return g.compile()


app = FastAPI(title="AI Trip Planner")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_frontend():
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "frontend", "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return {"message": "frontend/index.html not found"}


@app.get("/health")
def health():
    return {"status": "healthy", "service": "ai-trip-planner"}


# Enhanced Arize AX tracing initialization
tracer = None
if _TRACING:
    try:
        space_id = os.getenv("ARIZE_SPACE_ID")
        api_key = os.getenv("ARIZE_API_KEY")
        if space_id and api_key:
            # Register with Arize AX
            tp = register(
                space_id=space_id, 
                api_key=api_key, 
                project_name="ai-trip-planner",
                project_version="v1.0.0"
            )
            
            # Enhanced instrumentation for multi-agent system
            LangChainInstrumentor().instrument(
                tracer_provider=tp, 
                include_chains=True, 
                include_agents=True, 
                include_tools=True,
                include_retrievers=True,
                include_llms=True
            )
            LiteLLMInstrumentor().instrument(tracer_provider=tp, skip_dep_check=True)
            
            # Get tracer for custom spans
            tracer = trace.get_tracer("ai-trip-planner", "1.0.0")
            print("✅ Arize AX tracing initialized successfully")
        else:
            print("⚠️  Arize credentials not found. Set ARIZE_SPACE_ID and ARIZE_API_KEY environment variables.")
    except Exception as e:
        print(f"❌ Failed to initialize Arize tracing: {e}")
        tracer = None

@app.post("/plan-trip", response_model=TripResponse)
def plan_trip(req: TripRequest):
    # Generate unique session ID for this trip planning request
    session_id = str(uuid.uuid4())
    
    # Use session context for tracing related requests
    with using_session(session_id=session_id):
        if tracer and _TRACING:
            with tracer.start_as_current_span("trip_planning_workflow") as span:
                try:
                    # Add request attributes to span
                    span.set_attributes({
                        "trip.destination": req.destination,
                        "trip.duration": req.duration,
                        "trip.budget": req.budget or "not_specified",
                        "trip.interests": req.interests or "not_specified",
                        "session.id": session_id
                    })
                    
                    graph = build_graph()
                    # Only include necessary fields in initial state
                    # Agent outputs (research, budget, local, final) will be added during execution
                    state = {
                        "messages": [],
                        "trip_request": req.model_dump(),
                        "tool_calls": [],
                    }
                    
                    # Execute the multi-agent workflow
                    with tracer.start_as_current_span("langgraph_execution") as graph_span:
                        out = graph.invoke(state)
                        
                        # Add output metrics to span
                        graph_span.set_attributes({
                            "output.final_length": len(out.get("final", "")),
                            "output.tool_calls_count": len(out.get("tool_calls", [])),
                            "agents.research_completed": bool(out.get("research")),
                            "agents.budget_completed": bool(out.get("budget")),
                            "agents.local_completed": bool(out.get("local")),
                            "agents.final_completed": bool(out.get("final"))
                        })
                    
                    span.set_status(Status(StatusCode.OK))
                    return TripResponse(result=out.get("final", ""), tool_calls=out.get("tool_calls", []))
                    
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise e
        else:
            # Fallback without tracing
            graph = build_graph()
            state = {
                "messages": [],
                "trip_request": req.model_dump(),
                "tool_calls": [],
            }
            out = graph.invoke(state)
            return TripResponse(result=out.get("final", ""), tool_calls=out.get("tool_calls", []))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
