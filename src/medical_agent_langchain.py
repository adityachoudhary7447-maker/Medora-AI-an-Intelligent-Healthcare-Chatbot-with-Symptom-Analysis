"""
Streamlined Medical Agent with Proper Tool Flow
==============================================

This version ensures the agent:
1. Uses analyze_medical_image to extract visible symptoms
2. Uses analyze_symptoms_direct with the extracted symptoms  
3. Uses get_disease_description for the identified diseases
4. Provides clean, focused medical responses
"""

import json
import warnings
from typing import Optional

import pandas as pd
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

from medical_tools import get_enhanced_medical_tools, set_global_resources
from ollama_runtime import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_REQUEST_TIMEOUT,
    format_ollama_error,
)

warnings.filterwarnings('ignore')

class AgenticMedicalAssistant:
    """Streamlined medical assistant with proper tool workflow."""

    def __init__(self,
                 faiss_symptom_index: FAISS,
                 faiss_severity_index: FAISS,
                 df_disease_precautions: pd.DataFrame,
                 df_disease_symptom_description: pd.DataFrame,
                 df_disease_symptom_severity: pd.DataFrame,
                 vision_model: Optional[BaseChatModel] = None,
                 model_name: str = OLLAMA_MODEL,
                 use_memory: bool = True):

        print(f"🚀 Initializing AgenticMedicalAssistant with model: {model_name}")

        # Set global resources for tools
        set_global_resources(
            faiss_symptom_index, faiss_severity_index,
            df_disease_precautions, df_disease_symptom_description,
            df_disease_symptom_severity, vision_model
        )

        # Initialize LLM
        self.llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=model_name,
            temperature=0.1,
            num_ctx=4096,
            timeout=OLLAMA_REQUEST_TIMEOUT,
        )
        
        # Store vision model reference
        self.vision_model = vision_model
        self.has_vision = vision_model is not None
        print(f"👁️ Vision capabilities: {self.has_vision}")

        # Get enhanced tools
        self.tools = get_enhanced_medical_tools()
        print(f"🔧 Tools loaded: {[tool.name for tool in self.tools]}")

        # Streamlined system prompt focused on proper tool workflow
        system_prompt = """You are a medical assistant AI that uses specialized tools to analyze symptoms and provide medical information.

**CRITICAL: YOU MUST USE TOOL OUTPUTS EXACTLY AS PROVIDED**

**TOOL ARGUMENTS ARE REQUIRED AND MUST MATCH THESE NAMES:**
- analyze_symptoms_direct: pass the user's complete symptom text as `user_input`
- analyze_combined_symptoms: pass both `text_symptoms` and `visual_symptoms`
- analyze_symptom_severity: pass the symptom list as `symptoms_list`
- get_disease_description: pass the exact `disease` value returned by the analysis tool as `disease_name`
- get_disease_precautions: pass the exact `disease` value as `disease_name`
- analyze_medical_image: pass `additional_context` (or an empty string)
Never call a tool with missing arguments. Do not invent a disease name.

**CONVERSATION CONTINUITY - VERY IMPORTANT:**

When users have an ongoing conversation with previous image analysis:

1. **IF previous visual symptoms were identified AND user adds text symptoms:**
   - Use analyze_combined_symptoms(text_symptoms="user's new symptoms", visual_symptoms="previous visual symptoms")
   - This provides better diagnostic accuracy by combining all available information
   - DO NOT start a new analysis chain - build on existing findings

2. **IF this is the first image analysis request:**
   - Use analyze_medical_image() to extract visible symptoms from the image
   - Then use analyze_symptoms_direct() with the extracted visible symptoms

3. **IF user is only providing text symptoms (no image):**
   - Use analyze_symptoms_direct() with user's text input

**WORKFLOW FOR COMBINED ANALYSIS:**

When combining symptoms:
1. Use analyze_combined_symptoms(text_symptoms, visual_symptoms)
2. Use get_disease_description() for the top disease matches
3. Present results showing how both visual and text symptoms contributed

**CRITICAL FORMATTING RULES:**
- **ALWAYS use the exact text returned by get_disease_description() tool**
- **DO NOT add your own medical knowledge or descriptions**
- **DO NOT modify, summarize, or supplement tool outputs**
- **DO NOT write your own disease descriptions**

**MEDICAL SAFETY:**
- Never diagnose, infer a diagnosis, or present database matches as medical probabilities.
- Only name conditions returned by the symptom-analysis tool, and label them as database matches.
- If tool results do not support a response, say so and ask the user to consult a qualified professional.
- Include a reminder that this assistant does not replace professional medical advice.

**RESPONSE FORMAT (use tool outputs exactly):**
The local database returned these possible symptom matches (not a diagnosis):

1. **[Disease Name]**
   Database similarity match: [X]%
   
   [EXACT TEXT FROM get_disease_description() tool - DO NOT MODIFY]

2. **[Disease Name]**  
   Confidence: [X]%
   
   [EXACT TEXT FROM get_disease_description() tool - DO NOT MODIFY]

Would you like precautions for any of these conditions?

This information is not a diagnosis. Consult a qualified healthcare professional for medical advice.

**CONVERSATION MEMORY:**
- Remember previous image analysis results in the conversation
- Build upon previous findings when users add more symptoms
- Provide comprehensive analysis that considers all available information

**REMEMBER:**
- Trust the medical database tools completely
- Never substitute your own medical knowledge for tool outputs
- Always use the exact descriptions returned by tools
- Your job is to orchestrate tools and present results, building on conversation history"""

        # Create prompt template
        if use_memory:
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
        else:
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])

        # Create memory
        if use_memory:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        else:
            self.memory = None

        # Create agent
        self.agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )

        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,  # Keep verbose for debugging
            handle_parsing_errors=True,
            max_iterations=10,
            early_stopping_method="generate",
            return_intermediate_steps=True
        )

        print("✅ AgenticMedicalAssistant initialized successfully!")

    def chat(self, user_input: str) -> str:
        """
        Streamlined chat interface with proper tool workflow.
        
        Args:
            user_input: User's message/question
            
        Returns:
            Assistant's response
        """
        print(f"\n🎯 AGENT CHAT REQUEST: {user_input}")
        
        try:
            response = self.agent_executor.invoke({"input": user_input})
            
            # Log tool usage for debugging
            if 'intermediate_steps' in response:
                steps = response['intermediate_steps']
                print(f"🔍 Tools used: {len(steps)} steps")
                for i, step in enumerate(steps):
                    if hasattr(step, '__len__') and len(step) >= 2:
                        action, observation = step[0], step[1]
                        print(f"  {i+1}. {action.tool}")
                        # Show first 100 chars of observation for debugging
                        obs_preview = str(observation)[:100] + "..." if len(str(observation)) > 100 else str(observation)
                        print(f"     Result: {obs_preview}")
            
            final_output = response.get("output", "No output generated")
            print(f"✅ AGENT FINAL OUTPUT: {final_output[:200]}...")
            return final_output
            
        except Exception as e:
            if self._is_tool_call_error(e):
                fallback_response = self._grounded_tool_fallback(user_input)
                print("⚠️ Falling back to direct local medical tools after invalid tool call.")
                return fallback_response

            error_msg = format_ollama_error(e, self.llm.model)
            print(f"❌ Agent chat error: {error_msg}")
            return error_msg

    @staticmethod
    def _is_tool_call_error(error: Exception) -> bool:
        """Identify malformed or unsupported model tool calls, not runtime failures."""
        message = str(error).lower()
        return any(
            marker in message
            for marker in (
                "validation error for",
                "field required",
                "tool call",
                "tool_calls",
                "does not support tools",
                "could not parse tool",
            )
        )

    def _grounded_tool_fallback(self, user_input: str) -> str:
        """Run symptom matching and lookup tools directly if model tool calls fail."""
        try:
            tools_by_name = {medical_tool.name: medical_tool for medical_tool in self.tools}
            symptom_tool = tools_by_name["analyze_symptoms_direct"]
            description_tool = tools_by_name["get_disease_description"]
            analysis = json.loads(
                symptom_tool.invoke({"user_input": user_input})
            )
            disease_matches = analysis.get("diseases", [])
            if not analysis.get("success") or not disease_matches:
                return (
                    "I couldn't complete a reliable symptom match using the local "
                    "medical database. Please describe your symptoms clearly or try "
                    "again. I cannot provide a diagnosis. If this may be an emergency, "
                    "contact your local emergency service."
                )

            response_lines = [
                "The local medical database returned these possible symptom matches "
                "(not a diagnosis):"
            ]
            for match in disease_matches[:5]:
                disease_name = match.get("disease")
                if not isinstance(disease_name, str) or not disease_name.strip():
                    continue
                description = json.loads(
                    description_tool.invoke(
                        {"disease_name": disease_name}
                    )
                )
                response_lines.append(
                    f"\n- {disease_name.replace('_', ' ').title()}"
                )
                confidence = match.get("confidence_percentage")
                if isinstance(confidence, (int, float)):
                    response_lines.append(
                        f"  Database match score: {confidence:.1f}%"
                    )
                if description.get("success"):
                    response_lines.append(f"  {description['description']}")

            response_lines.extend([
                "",
                "These results are informational and do not establish a diagnosis. "
                "Please consult a qualified healthcare professional for medical "
                "advice. If symptoms are severe or urgent, contact your local "
                "emergency service.",
            ])
            return "\n".join(response_lines)
        except Exception as error:
            return (
                "I couldn't safely complete the local symptom analysis "
                f"({type(error).__name__}). Please try again or consult a qualified "
                "healthcare professional. If this may be an emergency, contact your "
                "local emergency service."
            )

    def reset_conversation(self):
        """Reset the conversation memory."""
        if self.memory is not None:
            self.memory.clear()

    def get_capabilities(self) -> dict:
        """Get information about assistant capabilities."""
        return {
            "text_analysis": True,
            "image_analysis": self.has_vision,
            "streamlined_workflow": True,
            "memory": self.memory is not None,
            "vision_model": str(self.vision_model.model) if self.vision_model else None,
            "available_tools": [tool.name for tool in self.tools]
        }


def create_medical_assistant(
    faiss_symptom_index: FAISS,
    faiss_severity_index: FAISS,
    df_disease_precautions: pd.DataFrame,
    df_disease_symptom_description: pd.DataFrame,
    df_disease_symptom_severity: pd.DataFrame,
    vision_model: Optional[BaseChatModel] = None,
    model_name: str = OLLAMA_MODEL,
    use_memory: bool = True
) -> AgenticMedicalAssistant:
    """
    Factory function to create a streamlined medical assistant.

    Args:
        faiss_symptom_index: FAISS index for symptom-disease matching
        faiss_severity_index: FAISS index for severity analysis
        df_disease_precautions: DataFrame with disease precautions
        df_disease_symptom_description: DataFrame with disease descriptions
        df_disease_symptom_severity: DataFrame with symptom severity scores
        vision_model: Optional local Ollama vision model for image analysis
        model_name: Local Ollama model name for text processing
        use_memory: Whether to use conversation memory

    Returns:
        Configured AgenticMedicalAssistant instance
    """
    print("🏭 Creating medical assistant...")
    
    return AgenticMedicalAssistant(
        faiss_symptom_index=faiss_symptom_index,
        faiss_severity_index=faiss_severity_index,
        df_disease_precautions=df_disease_precautions,
        df_disease_symptom_description=df_disease_symptom_description,
        df_disease_symptom_severity=df_disease_symptom_severity,
        vision_model=vision_model,
        model_name=model_name,
        use_memory=use_memory
    )