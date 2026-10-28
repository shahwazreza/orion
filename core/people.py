"""Meeting people: first-run onboarding, voice enrollment, and people commands.

Orion learns a name by asking, confirms it, asks consent before learning a
voice, and then collects ~10 s of natural speech. Only the speaker embeddings
are kept; the audio and what was said are discarded.
"""

from __future__ import annotations

import logging
import re

import numpy as np

from audio.speaker_id import identify
from core.dialog import Dialog, Heard, yes_no
from core.identity import Identity
from storage.users import User, UserStore

log = logging.getLogger(__name__)

# Natural prompts produce better voiceprints than reading fixed phrases.
VOICE_PROMPTS = [
    "First, tell me a little about your day so far.",
    "Lovely. What's a meal you could eat every week without getting bored?",
    "And if you could visit anywhere next year, where would you go, and why?",
    "One more. Describe the room you're in right now.",
    "Last one, I promise. What's something you're looking forward to?",
]

_NAME_LEAD = re.compile(
    r"^(?:(?:hi|hello|hey|well|so|um+|uh+|oh|yes|yeah|okay|ok)[, ]+)*"
    r"(?:my name is|my name's|the name's|the name is|i'm|i am|im|call me|it's|it is|this is|you can call me)\s+",
    re.I,
)
# The same phrases anywhere in the sentence: "Hey Orion, my name is Reza" (which
# Whisper may even hear as "Hey Ryan, ..."). Only the explicit ones, so "it's"
# in "it's nice, I'm Reza" doesn't grab the wrong words.
_NAME_ANYWHERE = re.compile(r"\b(?:my name is|my name's|the name's|the name is|call me|i'm|i am)\s+",
                            re.I)
_NOT_NAMES = {"yes", "no", "yeah", "nope", "okay", "ok", "hello", "hi", "hey", "thanks", "thank",
              "you", "sorry", "what", "nothing", "orion", "jarvis", "the", "a", "an",
              "bye", "goodbye", "cya", "later", "stop", "cancel", "never", "mind", "nevermind",
              "quit", "exit", "skip", "wait", "hmm", "um", "uh", "great", "cool", "fine", "good"}
# Said instead of a name: the person doesn't want to do introductions now.
_CANCEL_INTRO = re.compile(r"^\W*(bye|goodbye|stop|cancel|never ?mind|forget it|not now|"
                           r"no thanks|skip( it)?|later|quit)\W*$", re.I)


def extract_name(text: str) -> str | None:
    """'My name is reza.' -> 'Reza'. Returns None if it doesn't look like a name."""
    t = text.strip()
    leads = list(_NAME_ANYWHERE.finditer(t))
    t = t[leads[-1].end():] if leads else _NAME_LEAD.sub("", t)
    t = re.sub(r"[^A-Za-z' -]", " ", t)
    t = re.split(r"\b(?:and|but|nice|pleased|how)\b", t, maxsplit=1, flags=re.I)[0]
    words = [w.strip("'-") for w in t.split() if w.strip("'-")]
    if not 1 <= len(words) <= 3 or any(w.lower() in _NOT_NAMES for w in words):
        return None
    if any(len(w) < 2 for w in words) and len(words) == 1:
        return None
    return " ".join(w[:1].upper() + w[1:].lower() for w in words)


class People:
    # People need longer to answer setup questions than a normal follow-up.
    ANSWER_TIMEOUT_S = 10.0
    SHORT_ANSWER_S = 2.5    # shorter answers get one "could you say a bit more?"

    def __init__(self, store: UserStore, identity: Identity, enroll_target_s: float = 10.0,
                 min_answer_s: float = 1.0, duplicate_voice_score: float = 0.7,
                 verify_threshold: float = 0.55) -> None:
        self.store = store
        self.identity = identity
        self.enroll_target_s = enroll_target_s
        self.min_answer_s = min_answer_s
        self.duplicate_voice_score = duplicate_voice_score
        # Recognising someone uses identity.threshold (0.40), which lets about 1 in
        # 100 strangers through (scripts/voice_eval.py). Anything that changes who
        # Orion trusts needs this stronger match on the request itself.
        self.verify_threshold = verify_threshold
        self._returning: User | None = None   # set when a "new" person is a voiceless existing one

    # ---- onboarding -------------------------------------------------------
    def greet_on_start(self, dialog: Dialog) -> None:
        """First run: meet the owner. Later runs: offer a missing voiceprint."""
        if self.store.is_empty():
            if not self.store.persistent:
                dialog.say("Heads up: I can't reach the system keyring, so I'll forget "
                           "everyone when I restart.")
            dialog.say("Hello, I'm Orion. I don't think we've met.")
            user = self.meet(dialog, first=True)
            if user is not None:
                self.identity.set_current(user)
            return
        owner = self.store.owner()
        if dialog.hears_voice and owner and not owner.has_voice and not owner.voice_opt_out:
            if self.offer_voice(dialog, owner, intro=f"{owner.name}, I haven't learned your voice yet."):
                self.identity.set_current(owner)
                return
        # The keyboard is the owner's terminal; a voice could be anyone until recognised.
        dialog.say(f"Orion online. Hello, {owner.name}." if owner and not dialog.hears_voice
                   else "Orion online.")

    def meet(self, dialog: Dialog, first: bool = False, heard_name: str | None = None) -> User | None:
        """Learn a new person's name, then (with consent) their voice.

        ``heard_name`` is a name the person already offered ("my name is Reza").
        """
        name = self._ask_name(dialog, "What's your name?" if first else "Hello. What's your name?",
                              candidate=heard_name)
        if name is None:
            self._returning = None
            dialog.say("No problem. We can do introductions another time.")
            return None
        returning, self._returning = self._returning, None
        if returning is not None:
            dialog.say(f"Ah, {returning.name}, we've met. I just don't know your voice yet.")
            if dialog.hears_voice:
                self.enroll_voice(dialog, returning)
            return returning
        user = self.store.add(name)
        if dialog.hears_voice:
            self.offer_voice(dialog, user, intro=None)
        else:
            dialog.say(f"Lovely. I'll learn your voice, {name}, next time we talk out loud.")
        return user

    def _ask_name(self, dialog: Dialog, prompt: str, attempts: int = 3,
                  candidate: str | None = None) -> str | None:
        silent = 0
        last: str | None = None     # the name Orion most recently suggested
        for _ in range(attempts):
            if candidate is not None:
                name, candidate = candidate, None
            else:
                heard = dialog.ask(prompt, timeout_s=self.ANSWER_TIMEOUT_S)
                if heard is None:
                    silent += 1
                    if silent >= 2:
                        return None
                    prompt = "Sorry, I didn't hear anything. What's your name?"
                    continue
                # "Jimble, you got it right the first time": they're confirming the
                # name Orion already suggested, not giving a new one.
                if last and (last.lower() in heard.text.lower() or yes_no(heard.text) is True):
                    return last
                if _CANCEL_INTRO.match(heard.text):
                    return None
                name = extract_name(heard.text)
            if name is None:
                prompt = "Sorry, I didn't catch a name. Please say just your first name."
                continue
            existing = self.store.find(name)
            if existing is not None and not existing.has_voice and self._returning is None:
                # Someone Orion met before but never heard properly: same person, not a clash.
                self._returning = existing
                return name
            if existing is not None:
                prompt = f"I already know someone called {name}. What should I call you instead?"
                continue
            last = name
            answer = dialog.ask(f"Nice to meet you, {name}. Did I get your name right?",
                                timeout_s=self.ANSWER_TIMEOUT_S)
            if answer is not None and yes_no(answer.text) is True:
                return name
            prompt = "Sorry about that. Please say your name once more, slowly."
        return None

    def offer_voice(self, dialog: Dialog, user: User, intro: str | None) -> bool:
        """Ask consent, then enroll. Returns True if a voiceprint was saved."""
        question = ("Can I learn your voice, so I know it's you when you talk to me? "
                    "I keep a voice signature, never recordings, and you can ask me to forget it.")
        answer = dialog.ask(f"{intro} {question}" if intro else f"Thanks, {user.name}. {question}",
                            timeout_s=self.ANSWER_TIMEOUT_S)
        if answer is None or yes_no(answer.text) is not True:
            self.store.set_opt_out(user.id, True)
            dialog.say(f"Understood, {user.name}. If you change your mind, say: learn my voice.")
            return False
        return self.enroll_voice(dialog, user)

    def enroll_voice(self, dialog: Dialog, user: User) -> bool:
        dialog.say("Answer each question in a sentence or two, in your normal voice.")
        samples: list[np.ndarray] = []
        total = 0.0
        for prompt in VOICE_PROMPTS:
            if total >= self.enroll_target_s:
                break
            # One nudge per question: one-word answers ("Biryani.") carry too little voice.
            for attempt in (prompt, "Could you say a bit more about that?"):
                heard = dialog.ask(attempt, timeout_s=self.ANSWER_TIMEOUT_S)
                if heard is None or heard.audio is None:
                    break
                emb = self.identity.embed(heard.audio, heard.sample_rate)
                if emb is not None and heard.seconds >= self.min_answer_s:
                    samples.append(emb)
                    total += heard.seconds
                long_enough = heard.seconds >= self.SHORT_ANSWER_S
                del heard   # audio and text discarded right away
                if long_enough or total >= self.enroll_target_s:
                    break

        if total < self.enroll_target_s * 0.6:
            dialog.say(f"I didn't hear enough to learn your voice, {user.name}. "
                       "Say: learn my voice, whenever you'd like to try again.")
            return False

        centroid = np.mean(samples, axis=0)
        centroid /= np.linalg.norm(centroid) or 1.0
        others = [u for u in self.store.all() if u.id != user.id]
        clash = identify(centroid, others, threshold=self.duplicate_voice_score, margin=0.0).user
        if clash is not None:
            answer = dialog.ask(f"Your voice sounds a lot like {clash.name}'s. Are you sure you're "
                                f"not {clash.name}?")
            if answer is None or yes_no(answer.text) is not True:
                dialog.say("Okay, I won't save that voiceprint.")
                return False

        self.store.set_voice(user.id, samples)
        self.store.set_opt_out(user.id, False)
        dialog.say(f"Got it, {user.name}. I'll recognise your voice from now on.")
        return True

    # ---- commands ---------------------------------------------------------
    def _verified(self, dialog: Dialog, user: User) -> bool:
        """Is this request clearly from ``user``? Speaks a retry hint if not.

        The keyboard (text mode) is trusted. By voice, the request itself must
        match the user's voiceprint at ``verify_threshold``: being carried over
        from earlier in a conversation isn't enough.
        """
        if not dialog.hears_voice:
            return True
        c = self.identity.confidence
        if c is not None and c >= self.verify_threshold:
            return True
        log.info("Sensitive request not verified (confidence %s)",
                 "n/a" if c is None else f"{c:.2f}")
        dialog.say(f"For that I need to be sure it's you, {user.name}. Say hey jarvis, then "
                   "ask again in a full sentence, close to the mic.")
        return False

    def handle(self, heard: Heard, dialog: Dialog) -> bool:
        """Handle people-related commands. Returns True if the utterance was one."""
        t = heard.text.lower().strip().rstrip(".!?")
        user = self.identity.current

        if re.search(r"\b(who am i|what'?s my name|what is my name|do you know (who i am|my name|me))\b", t):
            if user:
                dialog.say(f"You're {user.name}.")
            elif dialog.hears_voice:
                dialog.say("I don't recognise your voice, I'm afraid.")
            else:
                dialog.say("I don't know yet.")
            return True

        if re.search(r"\b(who do you know|who have you met|who (all )?do you recogni[sz]e|"
                     r"list (the )?(users|people|profiles))\b", t):
            names = [u.name for u in self.store.all()]
            if not names:
                dialog.say("I haven't met anyone yet.")
            elif user is None and dialog.hears_voice:
                dialog.say(f"I know {len(names)} {'person' if len(names) == 1 else 'people'}.")
            else:
                dialog.say("I know " + _join(names) + ".")
            return True

        if re.search(r"\b((add|enrol+|register|meet|learn) (a |an )?(new |another |one more )?"
                     r"(person|user|voice|profile|member|someone|somebody|friend)"
                     r"|meet someone new)\b", t):
            if user is None or not user.is_owner:
                dialog.say("Only the owner can add new people. Ask them to say: add a new person.")
                return True
            if not self._verified(dialog, user):
                return True
            dialog.say("Of course. Could the new person come close and answer me?")
            new = self.meet(dialog)
            if new is not None:
                self.identity.set_current(new)
            return True

        m = re.search(r"\b(?:(?:re-?)?learn|re-?train|update|re-?enrol+) (\w[\w' -]*?)'s voice\b", t)
        if m and m.group(1) not in ("my", "your"):
            # The owner teaching Orion someone else's voice: the only way for a person
            # with no voiceprint yet, since Orion can't recognise them to let them ask.
            if user is None or not user.is_owner:
                dialog.say("Only the owner can do that.")
                return True
            if not self._verified(dialog, user):
                return True
            target = self.store.find(m.group(1))
            if target is None:
                dialog.say(f"I don't know anyone called {m.group(1).title()}.")
                return True
            dialog.say(f"Okay. {target.name}, come close and answer me.")
            if self.enroll_voice(dialog, target):
                self.identity.set_current(target)
            return True

        if re.search(r"\b((re-?)?learn|re-?train|update|re-?enrol+) my voice\b", t):
            if user is None:
                dialog.say("I don't recognise your voice yet, so I can't tell whose voice to "
                           "learn. Ask the owner to say: learn, your name's, voice.")
                return True
            if not self._verified(dialog, user):   # a lookalike could replace your voiceprint
                return True
            self.enroll_voice(dialog, user)
            return True

        m = re.match(r"^(?:please )?(?:call me|my name is|my name's|from now on call me)\s+(.+)$", t)
        if m:
            name = extract_name(m.group(1))
            if name is None:
                return False
            if user is None:
                if self.store.is_empty():
                    # Nobody set up yet (introductions were skipped): this person becomes the owner.
                    new = self.meet(dialog, first=True, heard_name=name)
                    if new is not None:
                        self.identity.set_current(new)
                    return True
                dialog.say(f"Nice to meet you, {name}. If you'd like me to remember you, "
                           "ask the owner to say: add a new person.")
                return True
            if name == user.name:
                dialog.say(f"I know, {name}.")
                return True
            if not self._verified(dialog, user):
                return True
            answer = dialog.ask(f"Should I call you {name} from now on, instead of {user.name}?")
            if answer is not None and yes_no(answer.text) is True:
                self.store.rename(user.id, name)
                dialog.say(f"Done. Hello, {name}.")
            else:
                dialog.say(f"Okay, I'll stick with {user.name}.")
            return True

        m = re.search(r"\b(?:delete|remove|forget) (?!my\b|me\b|who\b|that\b|it\b|everything\b)"
                      r"(\w[\w' -]*?)(?:'s (?:profile|voice(?:print)?))?$", t)
        if m and self.store.find(m.group(1)) is not None:
            target = self.store.find(m.group(1))
            if user is None or not user.is_owner:
                dialog.say("Only the owner can delete other people.")
                return True
            if target.id == user.id:
                m = None   # "delete Reza" by Reza: fall through to the self-delete flow
            elif not self._verified(dialog, user):
                return True
            else:
                answer = dialog.ask(f"Delete {target.name}'s name and voiceprint? Are you sure?",
                                    timeout_s=self.ANSWER_TIMEOUT_S)
                if answer is not None and yes_no(answer.text) is True:
                    self.store.delete(target.id)
                    dialog.say(f"Done. I've forgotten {target.name}.")
                else:
                    dialog.say(f"Okay, I've kept {target.name}.")
                return True

        if re.search(r"\b(delete|remove|forget) (my (profile|voice(print)?|account)|me|who i am)\b", t) or (
                m is not None and user is not None and self.store.find(m.group(1)) is not None
                and self.store.find(m.group(1)).id == user.id):
            if user is None:
                dialog.say("I don't recognise your voice, so there's nothing to delete.")
                return True
            if not self._verified(dialog, user):
                return True
            answer = dialog.ask(f"This deletes your name and voiceprint, {user.name}. "
                                "Are you sure?")
            if answer is not None and yes_no(answer.text) is True:
                self.store.delete(user.id)
                self.identity.set_current(None)
                dialog.say("Done. I've forgotten you. It was nice knowing you.")
            else:
                dialog.say("Okay, I've kept everything.")
            return True

        return False


def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
