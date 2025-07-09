#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""
Defines the structure of a set of preferences.
"""
import copy
import json
import typing as t
from types import SimpleNamespace
from preference_inferrer.preference_sets.abstract_preference_set import PreferenceSet

"""
Preference Mapping to how it's shown in paper. Left side is keys used below, right side is how it's show in the paper.

Summarization:
slf5k - reddit posts
ccn_dailymail - news articles
wikipedia - wikipedia pages
CShorten/ML-ArXiv-Papers - paper abstract
imdb - movie review

Email writing
slf5k - personal problem
ampere - paper review
paper_tweet - paper tweet
ccby - paper summary
"""


EMAIL_SOURCE_TO_PREFERENCE = {
    "plume": {
        "ccby": "be highly inquisitive, include several long and flowing sentences, "
                "use emojis, write using conditional expressions",
        "slf5k": "be intensely emotional, include alliterations, "
                 "sign off the email using an epithet, write using a stream-of-consciousness style",
        "ampere": "be sharply critical, include several short and punchy sentences, "
                  "use parenthetical asides, write using assertive expressions",
        "paper_tweet": "be blatantly sarcastic, include hyperboles, "
                       "open the email using a movie reference, write using bullet points",
    },
    "prelude": {
        "ccby": "professional greeting and closing, respectful, straight to the points, structured",
        "slf5k": "conversational, informal, no closing",
        "ampere": "call to action, casual tone, clear, positive",
        "paper_tweet": "engaging, personalized, professional tone, thankful closing",
    }
}

for fw, value in EMAIL_SOURCE_TO_PREFERENCE.items():
    for source, compound_preferences in value.items():
        EMAIL_SOURCE_TO_PREFERENCE[fw][source] = compound_preferences.replace(",", ";")

ARTICLE_SOURCE_TO_PREFERENCE = {
    "plume": {
        "cnn_dailymail": "adopt a step-by-step structure, include a simile, "
                         "use ampersands (&) instead of \"and\"s, write in the style of a children's book",
        "slf5k": "adopt a third person narrative, include rhetorical questions, "
                 "use ALLCAPS to emphasize certain words, write in the style of a tweet",
        "wikipedia": "adopt a rhyming structure, include modern slang, "
                     "use semicolons (;) when possible, write in the style of a screenplay",
        "CShorten/ML-ArXiv-Papers": "adopt a question-answering style structure, include personifications, "
                                    "use archaic language, write in the style of a podcast",
        "imdb": "adopt a second person narrative, include onomatopoeias, "
                "use imagery, write in the style of old timey radio",
    },
    "prelude": {
        "cnn_dailymail": "interactive, playful language, positive, "
                         "short sentences, storytelling, style targeted to young children",
        "slf5k": "brief, immersive, invoke personal reflection, second person narrative, show emotions",
        "wikipedia": "brief, bullet points, parallel structure",
        "CShorten/ML-ArXiv-Papers": "inquisitive, simple English, skillful foreshadowing, tweet style, with emojis",
        "imdb": "question answering style",
    }
}

for fw, value in ARTICLE_SOURCE_TO_PREFERENCE.items():
    for source, compound_preferences in value.items():
        ARTICLE_SOURCE_TO_PREFERENCE[fw][source] = compound_preferences.replace(",", ";")


class WritingPreferenceSet(PreferenceSet):
    """
    Defines a set of preferences, and a number of utility function around preference sets.
    Notes:
    - Because unbounded nature of natural language preferences in the user edit environment, we define a preference set
      as a list of preferences
    """

    def __init__(self, preferences: t.Union[None, str, t.Sequence[str]], config: SimpleNamespace):
        """
        Args:
            preferences: set of preference strings
        """
        super().__init__(config)
        if preferences is None:
            preferences = list()
        if isinstance(preferences, str):
            preferences = [preferences]
        self.preferences = preferences

    def in_natural_language(self, mode: str) -> t.Union[None, str, t.List[str], t.List[None]]:
        """
        Returns the set of preferences in natural language.
        # NOTE: Sorts preferences to make things deterministic in llm prompts
        Args:
            mode: One of "list", "string", or "json".
                  If "list", return a list of strings where each item describes one preference.
                  If "string", returns a single string that contains all preferences.
                  If "json", return a json string representation (output of json.dumps)
        Returns:
            Preferences in natural language
        """
        if len(self.preferences) == 0:
            if mode == "json":
                return json.dumps([])
            elif mode == "list":
                return []
            elif mode == "string":
                return None
            else:
                raise ValueError(f"Unknown mode: {mode}, must be one of 'json', 'list' or 'string'")
        else:
            preferences = list(self.preferences)
            if mode == "json":
                return json.dumps(preferences)
            elif mode == "list":
                return preferences
            elif mode == "string":
                return "; ".join(preferences)
            else:
                raise ValueError(f"Unknown mode: {mode}, must be one of 'json', 'list' or 'string'")

    def remove(self, preference: str):
        """
        Removes a preference from the set
        Args:
            preference: The preference to remove.
        """
        if preference in self.preferences:
            self.preferences.remove(preference)

    def update(self, other: "WritingPreferenceSet"):
        """
        Update the given preference set with a new preference set
        Args:
            other: the other preference set to update this preference set with

        """
        if isinstance(self.preferences, set):
            self.preferences.update(other.preferences)
        else:
            for preference_str in other.preferences:
                if preference_str not in self.preferences:
                    self.preferences.append(preference_str)

    def __eq__(self, other: "WritingPreferenceSet") -> bool:
        """
        Define the "=" operator to match set behavior
        Args:
            other: the other preference set to compare equality to
        Returns:
            True if preferred and dispreferred attributes are equal, otherwise False
        """
        return self.preferences == other.preferences

    def __sub__(self, other: "WritingPreferenceSet") -> bool:
        """
        Define the subtract (-) operator to match set behavior.
        Args:
            other: the other preference set to subtract
        Returns:
            A new preference set object containing the current preferences minus the "other" preferences
        """
        copy_of_self = copy.deepcopy(self)
        for preference in other.preferences:
            copy_of_self.remove(preference)
        return copy_of_self

    def __contains__(self, item) -> bool:
        return item in self.preferences

    def __len__(self) -> int:
        return len(self.preferences)

    def is_empty(self) -> bool:
        """
        Returns:
            True if no preferences are contained, else False
        """
        return len(self.preferences) == 0

    def get_number_of_contradictions(self) -> int:
        """
        Calculates the number of contradictions in this preference set.
        Contradictions are defined as an attribute that is both preferred and dispreferred.
        Returns:
            the number of contradictions
        """
        return 0

    def as_single_set(self) -> list:
        """
        Combines the preferred_attributes and dispreferred_attributes into a single set by prefixing them with
        appropriate strings.
        Returns:
            A single set describing the full preference set
        """
        return self.preferences

    @staticmethod
    def create_empty_preference_set(config: SimpleNamespace) -> "WritingPreferenceSet":
        """
        Generate an empty set of preferences.
        Args:
            config: configurations
        Returns:
             An empty preference set
        """
        return WritingPreferenceSet(list(), config)
