#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Memory System with Short-Term and Long-Term Memory
Uses gemini-flash-latest for memory processing with structured outputs
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from google import genai
from google.genai import types


class ImportanceLevel(str, Enum):
    """Importance levels for memory items"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MemoryCategory(str, Enum):
    """Categories for organizing memories"""
    PERSONAL_INFO = "personal_info"
    PREFERENCE = "preference"
    FACT = "fact"
    RELATIONSHIP = "relationship"
    GOAL = "goal"
    EXPERIENCE = "experience"
    SKILL = "skill"
    OTHER = "other"


@dataclass
class ConversationTurn:
    """Represents a single conversation turn"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    metadata: Dict[str, Any] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class MemoryItem:
    """Represents a single memory item"""
    content: str
    category: str  # MemoryCategory
    importance: str  # ImportanceLevel
    timestamp: str
    context: str = ""  # Additional context about when/how this was learned
    related_memories: List[str] = None  # IDs of related memories
    
    def to_dict(self):
        result = asdict(self)
        if self.related_memories is None:
            result['related_memories'] = []
        return result


class ShortTermMemory:
    """
    Manages recent conversation history and temporary context
    """
    
    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self.turns: List[ConversationTurn] = []
        self.logger = logging.getLogger('ShortTermMemory')
    
    def add_turn(self, role: str, content: str, metadata: Dict = None):
        """Add a conversation turn to short-term memory"""
        turn = ConversationTurn(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        self.turns.append(turn)
        
        # Keep only the most recent turns
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]
    
    def get_recent_conversation(self, num_turns: int = None) -> List[ConversationTurn]:
        """Get recent conversation turns"""
        if num_turns is None:
            return self.turns
        return self.turns[-num_turns:]
    
    def clear_processed_turns(self, num_turns: int):
        """Clear the oldest N processed turns"""
        if num_turns >= len(self.turns):
            self.turns = []
        else:
            self.turns = self.turns[num_turns:]
    
    def get_conversation_text(self, num_turns: int = None) -> str:
        """Get conversation as formatted text for processing"""
        turns = self.get_recent_conversation(num_turns)
        lines = []
        for turn in turns:
            lines.append(f"{turn.role.upper()}: {turn.content}")
        return "\n".join(lines)
    
    def is_ready_for_processing(self, min_turns: int = 5) -> bool:
        """Check if we have enough turns to warrant processing"""
        return len(self.turns) >= min_turns


class LongTermMemory:
    """
    Manages persistent, important memories
    """
    
    def __init__(self, memory_file: str = "long_term_memories.json"):
        self.memory_file = Path(memory_file)
        self.memories: Dict[str, List[MemoryItem]] = self._load_memories()
        self.logger = logging.getLogger('LongTermMemory')
    
    def _load_memories(self) -> Dict[str, List[MemoryItem]]:
        """Load memories from JSON file"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    # Convert back to MemoryItem objects
                    memories = {}
                    for category, items in data.items():
                        memories[category] = [
                            MemoryItem(**item) for item in items
                        ]
                    return memories
            except json.JSONDecodeError:
                self.logger.warning(f"Could not parse {self.memory_file}, starting fresh")
                return self._default_memories()
        return self._default_memories()
    
    def _default_memories(self) -> Dict[str, List[MemoryItem]]:
        """Default memory structure"""
        return {category.value: [] for category in MemoryCategory}
    
    def save_memories(self):
        """Save memories to JSON file"""
        # Convert MemoryItem objects to dicts
        data = {}
        for category, items in self.memories.items():
            data[category] = [item.to_dict() for item in items]
        
        with open(self.memory_file, 'w') as f:
            json.dump(data, f, indent=2)
        self.logger.info("Long-term memories saved")
    
    def add_memory(self, memory: MemoryItem):
        """Add a memory to long-term storage"""
        category = memory.category
        if category not in self.memories:
            self.memories[category] = []
        
        self.memories[category].append(memory)
        self.save_memories()
        self.logger.info(f"Added memory to category '{category}': {memory.content[:50]}...")
    
    def add_memories_batch(self, memories: List[MemoryItem]):
        """Add multiple memories at once"""
        for memory in memories:
            category = memory.category
            if category not in self.memories:
                self.memories[category] = []
            self.memories[category].append(memory)
        
        self.save_memories()
        self.logger.info(f"Added {len(memories)} memories to long-term storage")
    
    def get_all_memories(self) -> Dict[str, List[MemoryItem]]:
        """Get all memories organized by category"""
        return self.memories
    
    def get_memories_by_category(self, category: str) -> List[MemoryItem]:
        """Get memories for a specific category"""
        return self.memories.get(category, [])
    
    def get_important_memories(self, min_importance: ImportanceLevel = ImportanceLevel.MEDIUM) -> List[MemoryItem]:
        """Get all memories above a certain importance threshold"""
        importance_order = {
            ImportanceLevel.LOW: 0,
            ImportanceLevel.MEDIUM: 1,
            ImportanceLevel.HIGH: 2,
            ImportanceLevel.CRITICAL: 3
        }
        min_level = importance_order[min_importance]
        
        important = []
        for category, items in self.memories.items():
            for item in items:
                if importance_order.get(ImportanceLevel(item.importance), 0) >= min_level:
                    important.append(item)
        
        return important
    
    def format_for_context(self, max_items_per_category: int = 5) -> str:
        """Format memories for injection into conversation context"""
        lines = ["=== LONG-TERM MEMORY ===\n"]
        
        # Priority order for categories
        priority_categories = [
            MemoryCategory.PERSONAL_INFO,
            MemoryCategory.PREFERENCE,
            MemoryCategory.RELATIONSHIP,
            MemoryCategory.GOAL,
            MemoryCategory.FACT,
            MemoryCategory.SKILL,
            MemoryCategory.EXPERIENCE,
            MemoryCategory.OTHER
        ]
        
        for category in priority_categories:
            items = self.get_memories_by_category(category.value)
            if not items:
                continue
            
            # Sort by importance and recency
            items_sorted = sorted(
                items,
                key=lambda x: (
                    {"critical": 3, "high": 2, "medium": 1, "low": 0}.get(x.importance, 0),
                    x.timestamp
                ),
                reverse=True
            )
            
            # Take top items
            top_items = items_sorted[:max_items_per_category]
            
            if top_items:
                lines.append(f"\n{category.value.upper().replace('_', ' ')}:")
                for item in top_items:
                    lines.append(f"  - {item.content}")
        
        lines.append("\n=== END LONG-TERM MEMORY ===")
        return "\n".join(lines)


class MemoryProcessor:
    """
    Processes short-term memory to extract important information
    Uses gemini-flash-latest with structured outputs
    """
    
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.logger = logging.getLogger('MemoryProcessor')
        
        # Define structured output schema for memory extraction
        self.memory_extraction_schema = {
            "type": "object",
            "properties": {
                "memories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The actual memory content - a fact, preference, or information to remember"
                            },
                            "category": {
                                "type": "string",
                                "enum": [cat.value for cat in MemoryCategory],
                                "description": "Category of the memory"
                            },
                            "importance": {
                                "type": "string",
                                "enum": [imp.value for imp in ImportanceLevel],
                                "description": "Importance level: critical (core identity/crucial facts), high (important preferences/goals), medium (useful facts), low (minor details)"
                            },
                            "context": {
                                "type": "string",
                                "description": "Brief context about when/how this was learned"
                            }
                        },
                        "required": ["content", "category", "importance"]
                    }
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of which memories were extracted and why"
                }
            },
            "required": ["memories", "reasoning"]
        }
    
    async def process_conversation(
        self, 
        conversation_text: str,
        existing_memories_context: str = ""
    ) -> List[MemoryItem]:
        """
        Process conversation text and extract important memories
        Returns list of MemoryItem objects
        """
        self.logger.info("Processing conversation for memory extraction...")
        
        # Build prompt for memory extraction
        prompt = self._build_extraction_prompt(conversation_text, existing_memories_context)
        
        try:
            # Use gemini-flash-latest for memory processing
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model="gemini-flash-latest",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=self.memory_extraction_schema,
                    temperature=0.3,  # Lower temperature for more consistent extraction
                )
            )
            
            # Parse the structured response
            result = json.loads(response.text)
            
            self.logger.info(f"Memory extraction reasoning: {result.get('reasoning', 'N/A')}")
            
            # Convert to MemoryItem objects
            memories = []
            for mem_data in result.get("memories", []):
                memory = MemoryItem(
                    content=mem_data["content"],
                    category=mem_data["category"],
                    importance=mem_data["importance"],
                    timestamp=datetime.now().isoformat(),
                    context=mem_data.get("context", ""),
                    related_memories=[]
                )
                memories.append(memory)
            
            self.logger.info(f"Extracted {len(memories)} memories from conversation")
            return memories
            
        except Exception as e:
            self.logger.error(f"Error processing conversation: {e}")
            return []
    
    def _build_extraction_prompt(self, conversation_text: str, existing_memories: str) -> str:
        """Build the prompt for memory extraction"""
        prompt = f"""You are a memory extraction system. Your job is to analyze conversations and extract important, memorable information about the user.

EXISTING MEMORIES:
{existing_memories if existing_memories else "No existing memories yet."}

RECENT CONVERSATION:
{conversation_text}

TASK:
Analyze the conversation and extract memories that meet these criteria:

IMPORTANCE GUIDELINES:
- CRITICAL: Core identity (name, occupation, location), life-changing goals, critical health info, deep values/beliefs
- HIGH: Strong preferences, ongoing goals/projects, important relationships, significant experiences, key skills
- MEDIUM: Moderate preferences, useful facts, casual interests, general information
- LOW: Minor details, one-off mentions, trivial information

EXTRACTION RULES:
1. Only extract NEW information not already in existing memories (avoid duplicates)
2. Focus on information about the USER, not general facts about the world
3. Extract concrete, specific information (avoid vague statements)
4. Prioritize information that reveals personality, preferences, or important life context
5. Skip pleasantries, small talk, and trivial exchanges
6. If the user corrects previous information, extract the correction
7. Combine related information into coherent memory items

CATEGORIES:
- personal_info: Name, age, occupation, location, family, identity
- preference: Likes, dislikes, habits, communication style
- fact: Concrete facts the user shared (events, possessions, situations)
- relationship: Information about people in the user's life
- goal: Aspirations, objectives, things they want to accomplish
- experience: Past experiences, stories, significant events
- skill: Abilities, expertise, knowledge areas
- other: Anything else important that doesn't fit above

Return ONLY the memories you extract. If nothing important to remember, return an empty memories array."""

        return prompt
    
    async def consolidate_memories(
        self,
        memories: List[MemoryItem]
    ) -> List[MemoryItem]:
        """
        Consolidate and deduplicate memories
        Uses LLM to merge similar memories and remove redundancy
        """
        if len(memories) < 5:
            return memories  # Not worth consolidating small sets
        
        self.logger.info(f"Consolidating {len(memories)} memories...")
        
        # Format memories for processing
        memories_text = "\n".join([
            f"{i+1}. [{mem.category}] ({mem.importance}) {mem.content}"
            for i, mem in enumerate(memories)
        ])
        
        consolidation_schema = {
            "type": "object",
            "properties": {
                "consolidated_memories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "category": {"type": "string", "enum": [cat.value for cat in MemoryCategory]},
                            "importance": {"type": "string", "enum": [imp.value for imp in ImportanceLevel]},
                            "context": {"type": "string"}
                        },
                        "required": ["content", "category", "importance"]
                    }
                },
                "reasoning": {"type": "string"}
            },
            "required": ["consolidated_memories"]
        }
        
        prompt = f"""You are a memory consolidation system. Review these memories and consolidate them by:
1. Merging duplicate or highly similar memories
2. Combining related facts into coherent statements
3. Removing redundancy while preserving all unique information
4. Maintaining or upgrading importance levels appropriately

MEMORIES TO CONSOLIDATE:
{memories_text}

Return the consolidated set of memories. Preserve all unique information."""

        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model="gemini-flash-latest",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=consolidation_schema,
                    temperature=0.2,
                )
            )
            
            result = json.loads(response.text)
            
            consolidated = []
            for mem_data in result.get("consolidated_memories", []):
                memory = MemoryItem(
                    content=mem_data["content"],
                    category=mem_data["category"],
                    importance=mem_data["importance"],
                    timestamp=datetime.now().isoformat(),
                    context=mem_data.get("context", ""),
                    related_memories=[]
                )
                consolidated.append(memory)
            
            self.logger.info(f"Consolidated to {len(consolidated)} memories")
            return consolidated
            
        except Exception as e:
            self.logger.error(f"Error consolidating memories: {e}")
            return memories  # Return original on error


class IntegratedMemorySystem:
    """
    Integrates short-term and long-term memory with automatic processing
    """
    
    def __init__(self, api_key: str, stm_max_turns: int = 20, process_threshold: int = 8):
        self.short_term = ShortTermMemory(max_turns=stm_max_turns)
        self.long_term = LongTermMemory()
        self.processor = MemoryProcessor(api_key)
        
        self.process_threshold = process_threshold  # Process STM after this many turns
        self.turns_since_processing = 0
        
        self.logger = logging.getLogger('IntegratedMemorySystem')
    
    def add_interaction(self, role: str, content: str, metadata: Dict = None):
        """Add an interaction to short-term memory"""
        # Skip empty or very short interactions
        if not content or len(content.strip()) < 3:
            return
        
        # Skip purely audio references
        if content.startswith("[Audio") and content.endswith("]"):
            return
        
        self.short_term.add_turn(role, content, metadata)
        self.turns_since_processing += 1
    
    async def process_if_ready(self, force: bool = False) -> bool:
        """
        Process short-term memory if ready, moving important items to long-term
        Returns True if processing occurred
        """
        if not force and self.turns_since_processing < self.process_threshold:
            return False
        
        if not self.short_term.is_ready_for_processing(min_turns=3):
            return False
        
        self.logger.info("Processing short-term memory...")
        
        # Get conversation text
        conversation_text = self.short_term.get_conversation_text()
        
        # Get existing memories for context (avoid duplicates)
        existing_context = self.long_term.format_for_context(max_items_per_category=10)
        
        # Extract memories
        new_memories = await self.processor.process_conversation(
            conversation_text,
            existing_context
        )
        
        # Filter for meaningful memories (medium importance and above)
        important_memories = [
            mem for mem in new_memories
            if mem.importance in [
                ImportanceLevel.MEDIUM.value,
                ImportanceLevel.HIGH.value,
                ImportanceLevel.CRITICAL.value
            ]
        ]
        
        if important_memories:
            # Add to long-term memory
            self.long_term.add_memories_batch(important_memories)
            self.logger.info(f"Moved {len(important_memories)} important memories to long-term storage")
        
        # Clear processed turns from short-term memory
        num_to_clear = len(self.short_term.turns) // 2  # Keep half for context
        self.short_term.clear_processed_turns(num_to_clear)
        
        self.turns_since_processing = 0
        return True
    
    def get_context_for_model(self) -> str:
        """Get formatted memory context to inject into model's system instruction"""
        ltm_context = self.long_term.format_for_context()
        
        # Optionally include recent STM for immediate context
        recent_stm = self.short_term.get_conversation_text(num_turns=5)
        
        if recent_stm:
            return f"{ltm_context}\n\n=== RECENT CONTEXT ===\n{recent_stm}\n=== END RECENT CONTEXT ==="
        
        return ltm_context
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        total_ltm = sum(len(items) for items in self.long_term.memories.values())
        
        return {
            "short_term_turns": len(self.short_term.turns),
            "turns_since_processing": self.turns_since_processing,
            "long_term_memories": total_ltm,
            "memories_by_category": {
                cat: len(items) for cat, items in self.long_term.memories.items()
            }
        }

