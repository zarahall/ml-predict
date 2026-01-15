"""
Monkey-patch for PROSE to use constrained vocabulary.
Import this before running ml-predict experiments for fair comparison.

Patches ALL prompt methods that could generate preferences:
- refine_preferences_prompt (main refinement - abstract_templates.py)
- basis_instruction_prompt (breakdown)
- coalesce_prompt (combine)
- get_preference_inference_prompt (initial inference - plume_templates.py)
"""

# Available traits (40 total, matching steering pretrained vectors)
CONSTRAINED_TRAITS = ['allcaps_emphasis', 'alliteration', 'ampersand_usage', 'archaic_language', 'bullet_parallel', 'childlike', 'conditional_expressions', 'critical', 'email_epithet_signoff', 'emoji_usage', 'formal_tone', 'header_structured', 'hyperbole', 'inquisitive', 'intensely_emotional', 'interactive_playful', 'interpretive', 'long_flowing_sentences', 'modern_slang', 'old_timey_radio', 'onomatopoeia', 'open_with_movie_ref', 'parenthetical_asides', 'personification', 'podcast_style', 'question_answering_style', 'rhetorical_questions', 'rhyming_screenplay', 'rhyming_structure', 'sarcastic', 'screenplay', 'second_person_narrative', 'semicolon_usage', 'short_punchy_sentences', 'simile_usage', 'step_by_step', 'stream_consciousness', 'third_person_perspective', 'tweet_style', 'vivid_imagery']

TRAIT_LIST = ', '.join(CONSTRAINED_TRAITS)

# Few-shot examples for better instruction following
VOCAB_CONSTRAINT = f"""

=== CRITICAL OUTPUT FORMAT CONSTRAINT ===

ALLOWED TRAIT NAMES (pick 1-3 from this list ONLY):
{TRAIT_LIST}

OUTPUT FORMAT: JSON array with trait names from the list above.

EXAMPLES OF CORRECT OUTPUT:
- User writes in a childlike style with similes → Preferences: ["childlike", "simile_usage"]
- User uses rhetorical questions and sarcasm → Preferences: ["rhetorical_questions", "sarcastic"]
- User writes formal emails with bullet points → Preferences: ["formal_tone", "bullet_parallel"]
- User writes vivid emotional content → Preferences: ["vivid_imagery", "intensely_emotional"]
- User writes step-by-step with hyperbole → Preferences: ["step_by_step", "hyperbole"]

EXAMPLES OF WRONG OUTPUT (DO NOT DO THIS):
- WRONG: "use storytelling techniques and analogies" (too long, not from list)
- WRONG: "childlike; simile_usage" (wrong format, use JSON array)
- WRONG: "child-like style" (not exact trait name from list)

Your output MUST be: Preferences: ["trait1", "trait2"] or Preferences: ["trait1", "trait2", "trait3"]
"""

def apply_prose_vocabulary_constraint():
    """Apply vocabulary constraint to ALL PROSE prompts."""
    patched_count = 0

    # PATCH 1: Abstract Templates (base class - has refine_preferences_prompt)
    try:
        from preference_inferrer.prompt_templates import abstract_templates
        Templates = abstract_templates.Templates

        # Patch refine_preferences_prompt (the main refinement method)
        if not hasattr(Templates, '_original_refine_preferences_prompt'):
            Templates._original_refine_preferences_prompt = Templates.refine_preferences_prompt

            def constrained_refine_preferences_prompt(self, preferences, user_traj, counterfactual=None):
                original = self._original_refine_preferences_prompt(preferences, user_traj, counterfactual)
                return original + VOCAB_CONSTRAINT

            Templates.refine_preferences_prompt = constrained_refine_preferences_prompt
            print("  [1/4] Patched Templates.refine_preferences_prompt")
            patched_count += 1

        # Patch basis_instruction_prompt
        if not hasattr(Templates, '_original_basis_instruction_prompt'):
            Templates._original_basis_instruction_prompt = Templates.basis_instruction_prompt

            @staticmethod
            def constrained_basis_instruction_prompt(raw_preference):
                original = abstract_templates.Templates._original_basis_instruction_prompt(raw_preference)
                return original + VOCAB_CONSTRAINT

            Templates.basis_instruction_prompt = constrained_basis_instruction_prompt
            print("  [2/4] Patched Templates.basis_instruction_prompt")
            patched_count += 1

        # Patch coalesce_prompt
        if not hasattr(Templates, '_original_coalesce_prompt'):
            Templates._original_coalesce_prompt = Templates.coalesce_prompt

            def constrained_coalesce_prompt(self, preferences):
                original = self._original_coalesce_prompt(preferences)
                return original + VOCAB_CONSTRAINT

            Templates.coalesce_prompt = constrained_coalesce_prompt
            print("  [3/4] Patched Templates.coalesce_prompt")
            patched_count += 1

    except ImportError as e:
        print(f"Warning: Could not patch abstract_templates: {e}")

    # PATCH 2: Plume Templates (framework-specific)
    try:
        from preference_inferrer.prompt_templates import plume_templates
        PlumeTemplates = plume_templates.PlumeTemplates

        # Patch get_preference_inference_prompt
        if not hasattr(PlumeTemplates, '_original_get_preference_inference_prompt'):
            PlumeTemplates._original_get_preference_inference_prompt = PlumeTemplates.get_preference_inference_prompt

            def constrained_get_preference_inference_prompt(self, trajectories):
                original = self._original_get_preference_inference_prompt(trajectories)
                return original + VOCAB_CONSTRAINT

            PlumeTemplates.get_preference_inference_prompt = constrained_get_preference_inference_prompt
            print("  [4/4] Patched PlumeTemplates.get_preference_inference_prompt")
            patched_count += 1

    except ImportError as e:
        print(f"Warning: Could not patch plume_templates: {e}")

    if patched_count > 0:
        print(f"\n*** VOCABULARY CONSTRAINT ACTIVE (FEW-SHOT) ***")
        print(f"*** Patched {patched_count} prompt methods ***")
        print(f"*** PROSE will output JSON arrays: [\"trait1\", \"trait2\", \"trait3\"] ***\n")
        return True
    else:
        print("WARNING: No prompts were patched!")
        return False


# Auto-apply when imported
if __name__ != "__main__":
    print("\n=== Applying PROSE Vocabulary Constraint ===")
    apply_prose_vocabulary_constraint()
