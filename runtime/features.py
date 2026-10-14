"""Public-only UI grammar and fixed-shape features, shared verbatim with deployment."""
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
import re
import numpy as np

INVENTORY = 55
TTY_ROWS = 24
TOKENS = 32
VOCAB = 2048
TEXT_BYTES = 512
UI_FEATURES = 16
_WORDS = re.compile(rb"[+-]?\d+|[a-z][a-z0-9']*")
_MENU = re.compile(r"^\s*([a-zA-Z0-9$])\s+([-+])\s+(.+?)\s*$")


class Mode(IntEnum):
    PLAY = 0
    ITEM = 1
    DIRECTION = 2
    MENU = 3
    CONFIRM = 4
    MORE = 5
    RAW = 6


def public_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.split(b"\0", 1)[0].decode("latin1")
    return np.asarray(value, dtype=np.uint8).tobytes().split(b"\0", 1)[0].decode("latin1")


@lru_cache(maxsize=65536)
def tokenize(text):
    result = np.zeros(TOKENS, dtype=np.int16)
    for i, word in enumerate(_WORDS.findall(text.encode("latin1", errors="replace").lower())[:TOKENS]):
        h = 2166136261
        for b in word:
            h = ((h ^ b) * 16777619) & 0xFFFFFFFF
        result[i] = 1 + h % (VOCAB - 1)
    result.flags.writeable = False
    return result


@dataclass
class View:
    mode: Mode
    mask: np.ndarray
    primitive: np.ndarray
    inventory_text: list
    menu_text: list
    menu_selected: np.ndarray


class ActionCodec:
    def __init__(self, actions, cache=False, forbid_quit=False):
        self.forbid_quit = bool(forbid_quit)
        self.actions = tuple(int(a) for a in actions)
        self.lookup = {key: i for i, key in enumerate(self.actions)}
        self.raw_count = len(self.actions)
        self.size = self.raw_count + INVENTORY + TTY_ROWS
        self.previous = self.raw_count
        self.cache = bool(cache)
        self._tty_key = self._inventory_key = None

    def reset(self):
        self.previous = self.raw_count
        self._tty_key = self._inventory_key = None

    def _controls(self, mask, keys):
        for key in keys:
            if key in self.lookup:
                mask[self.lookup[key]] = 1

    def view(self, obs):
        message = public_text(obs["message"])
        tty = np.asarray(obs["tty_chars"], dtype=np.uint8)[:TTY_ROWS]
        tty_key = (tty.shape, tty.tobytes()) if self.cache else None
        # Encode the complete terminal, including menu headers/prompts outside
        # the map. Only recognized selectable rows receive pointer actions.
        if self.cache and tty_key == self._tty_key:
            lines, menu_text, selected, menu_keys = self._tty_value
        else:
            lines = [public_text(row).rstrip() for row in tty]
            menu_text = lines + [""] * (TTY_ROWS - len(lines))
            selected = np.zeros((TTY_ROWS, 1), dtype=np.float32)
            menu_keys = {}
            for row, line in enumerate(lines):
                match = _MENU.match(line)
                if match:
                    menu_keys[row] = ord(match[1])
                    menu_text[row] = match[3]
                    selected[row, 0] = float(match[2] == "+")
            if self.cache:
                self._tty_key, self._tty_value = tty_key, (lines, menu_text, selected, menu_keys)
        prompt = " ".join((message + " " + " ".join(lines[:2])).lower().split())
        raw_inventory = np.asarray(obs["inv_strs"], dtype=np.uint8)[:INVENTORY]
        inventory_key = (raw_inventory.shape, raw_inventory.tobytes()) if self.cache else None
        if self.cache and inventory_key == self._inventory_key:
            inventory = self._inventory_value
        else:
            inventory = [public_text(row) for row in raw_inventory]
            inventory += [""] * (INVENTORY - len(inventory))
            if self.cache:
                self._inventory_key, self._inventory_value = inventory_key, inventory
        misc = np.asarray(obs["misc"]).reshape(-1)
        more = bool(len(misc) > 2 and misc[2])
        yn = bool(len(misc) and misc[0])
        getline = bool(len(misc) > 1 and misc[1])
        if more and not menu_keys:
            mode = Mode.MORE
        elif yn and any(s in prompt for s in ("what direction", "which direction", "in what direction")):
            mode = Mode.DIRECTION
        elif yn and "what do you want to" in prompt and "?" in prompt:
            mode = Mode.ITEM
        elif menu_keys and more:
            mode = Mode.MENU
        elif yn and re.search(r"\[[^\]]*[yn][^\]]*\]", prompt):
            mode = Mode.CONFIRM
        elif yn or getline or menu_keys:
            mode = Mode.RAW
        else:
            mode = Mode.PLAY

        mask = np.zeros(self.size, dtype=np.int8)
        primitive = np.full(self.size, -1, dtype=np.int16)
        primitive[:self.raw_count] = np.arange(self.raw_count)
        escape = self.lookup.get(27, 0)
        if mode in (Mode.PLAY, Mode.RAW):
            mask[:self.raw_count] = 1
        elif mode == Mode.MORE:
            key = next((k for k in (32, 13, 10, 27) if k in self.lookup), self.actions[0])
            mask[self.lookup[key]] = 1
        elif mode == Mode.DIRECTION:
            self._controls(mask, map(ord, "hjklyubn><.5"))
            mask[escape] = 1
        elif mode == Mode.CONFIRM:
            self._controls(mask, (ord("y"), ord("n"), 27, 13, 10))
            for choices in re.findall(r"\[([a-zA-Z]+)\]", prompt):
                if "y" in choices and "n" in choices:
                    self._controls(mask, map(ord, choices))
        elif mode == Mode.ITEM:
            self._controls(mask, (27, 13, 10, ord("*"), ord("?"), ord("-"), ord("$")))
            used = set()
            for slot, letter in enumerate(np.asarray(obs["inv_letters"]).reshape(-1)[:INVENTORY]):
                action = self.lookup.get(int(letter))
                if action is not None and inventory[slot] and action not in used:
                    used.add(action)
                    index = self.raw_count + slot
                    primitive[index] = action
                    mask[index] = 1
                    mask[action] = 0
        elif mode == Mode.MENU:
            self._controls(mask, (27, 13, 10, 32, ord("<"), ord(">")))
            used = set()
            for row, letter in menu_keys.items():
                action = self.lookup.get(letter)
                if action is not None and action not in used:
                    used.add(action)
                    index = self.raw_count + INVENTORY + row
                    primitive[index] = action
                    mask[index] = 1
                    mask[action] = 0
        if self.forbid_quit:
            # Pinned NLE Command.QUIT == Meta-q (241); ordinary q is quaff.
            # Filter decoded primitives too, so pointer aliases cannot bypass it.
            quit_action = self.lookup.get(241)
            if quit_action is not None:
                mask[primitive == quit_action] = 0
            # Also covers #quit / any other route to the actual public prompt.
            # Gate on active yn UI, so a stale message never forces a reply.
            if yn and re.search(r"\breally quit\s*\?", prompt):
                mask[:] = 0
                self._controls(mask, (ord("n"),))
        if not mask.any():
            mask[escape] = 1
        return View(mode, mask, primitive, inventory, menu_text, selected)

    def decode(self, action, view):
        action = int(action)
        if not 0 <= action < self.size or not view.mask[action]:
            raise ValueError(f"invalid masked policy action {action} in mode {view.mode.name}")
        primitive = int(view.primitive[action])
        if not 0 <= primitive < self.raw_count:
            raise ValueError("codec produced a non-arena primitive")
        self.previous = primitive
        return primitive


class FeatureBuilder:
    def __init__(self, actions, glyph_count, cache=False, full_cache=False, architecture="pooled_v3", forbid_quit=False):
        self.codec = ActionCodec(actions, cache=cache, forbid_quit=forbid_quit)
        self.glyph_count = int(glyph_count)
        self.last_view = None
        self.cache = bool(cache)
        self._inventory_text = self._menu_text = None
        self.full_cache = bool(full_cache)
        self._full_key = self._full_value = None
        self.full_cache_hits = self.full_cache_misses = 0
        self.architecture = architecture
        self._reset_history()

    def _reset_history(self):
        self._last_public = None
        self._zero_streak = 0
        self._visited = set()
        self._initial_xp = self._max_xp = 0

    def _ui_history(self, obs):
        s = np.asarray(obs["blstats"])
        cursor = np.asarray(obs["tty_cursor"]).reshape(-1)
        row, col = map(int, cursor[:2])
        position = tuple(int(s[i]) for i in (23, 24, 0, 1))
        turn, xp = int(s[20]), int(s[19])
        previous = self._last_public
        delta = max(0, turn - previous[0]) if previous else 0
        self._zero_streak = self._zero_streak + 1 if previous and turn == previous[0] else 0
        if previous is None:
            self._initial_xp = self._max_xp = xp
            self._visited.add(position)
        elif delta:
            self._max_xp = max(self._max_xp, xp)
            if len(self._visited) < 257:
                self._visited.add(position)
        ui = np.zeros(UI_FEATURES, np.float32)
        ui[:3] = np.asarray(obs["misc"]).reshape(-1)[:3]
        ui[3:7] = row / 23, col / 79, (col - int(s[0])) / 79, (row - 1 - int(s[1])) / 23
        ui[7] = np.log1p(min(delta, 10000)) / np.log(10001)
        ui[8] = np.log1p(min(self._zero_streak, 1000000)) / np.log(1000001)
        ui[9] = int(self.last_view.mode) / 6 if self.last_view else 0
        if previous:
            ui[10:13] = (row - previous[1]) / 23, (col - previous[2]) / 79, float(position == previous[3])
            ui[15] = np.log1p(min(max(0, xp - previous[4]), 100)) / np.log(101)
        ui[13] = min(1, np.log1p(max(0, self._max_xp - self._initial_xp)) / np.log(101))
        ui[14] = (len(self._visited) - 1) / 256
        self._last_public = (turn, row, col, position, xp)
        return np.clip(ui, -1, 1)

    @staticmethod
    def _ordered_text(obs):
        # Fixed boundaries preserve byte order, punctuation and letter case.
        result = np.zeros(TEXT_BYTES, np.int16)
        message = public_text(obs["message"]).encode("latin1")[:256]
        tty = np.asarray(obs["tty_chars"], np.uint8)
        row = min(max(int(np.asarray(obs["tty_cursor"])[0]), 0), len(tty) - 1)
        for offset, data in ((0, message), (256, tty[:2].tobytes()[:160]), (416, tty[row].tobytes()[:80])):
            result[offset:offset + len(data)] = np.frombuffer(data, np.uint8).astype(np.int16) + 1
        return result

    def reset(self):
        self.codec.reset()
        self.last_view = None
        self._inventory_text = self._menu_text = None
        self._full_key = self._full_value = None
        self._reset_history()

    def prepare(self, obs):
        ui = self._ui_history(obs) if self.architecture == "cnn_ui_v1" else None
        key = None
        if self.full_cache:
            # Every public field consumed below or by ActionCodec.view is in
            # this exact byte key. Previous action is history, refreshed even
            # when the public observation is unchanged. Reset invalidates all.
            names = ("glyphs", "blstats", "message", "inv_strs", "inv_glyphs",
                     "inv_oclasses", "inv_letters", "tty_chars", "misc")
            if ui is not None:
                names += ("tty_cursor",)
            arrays = [np.asarray(obs[name]) for name in names]
            key = tuple((a.shape, a.dtype.str, a.tobytes()) for a in arrays)
            if key == self._full_key:
                self.full_cache_hits += 1
                result = dict(self._full_value)
                result["previous"] = np.array([self.codec.previous], dtype=np.int16)
                if ui is not None:
                    result["ui"] = ui
                return result
            self.full_cache_misses += 1
        view = self.codec.view(obs)
        self.last_view = view
        glyphs = np.asarray(obs["glyphs"], dtype=np.int16)
        if glyphs.shape != (21, 79):
            raise ValueError(f"unexpected public map shape {glyphs.shape}")
        stats = np.asarray(obs["blstats"], dtype=np.float32)
        x, y = int(stats[0]), int(stats[1])
        crop = np.full((9, 9), self.glyph_count, dtype=np.int16)
        x0, x1 = max(0, x - 4), min(79, x + 5)
        y0, y1 = max(0, y - 4), min(21, y + 5)
        crop[y0 - y + 4:y1 - y + 4, x0 - x + 4:x1 - x + 4] = glyphs[y0:y1, x0:x1]
        transformed = np.clip(np.sign(stats) * np.log1p(np.abs(stats)), -10, 10) / 10
        condition = int(np.asarray(obs["blstats"])[25])
        bits = np.array([(condition >> i) & 1 for i in range(32)], dtype=np.float32)
        inv_glyphs = np.asarray(obs["inv_glyphs"], dtype=np.int16).copy()
        inv_glyphs[inv_glyphs < 0] = self.glyph_count
        inv_glyphs = np.minimum(inv_glyphs, self.glyph_count)
        inv_aux = np.stack((np.asarray(obs["inv_oclasses"], dtype=np.float32) / 16,
                            np.asarray(obs["inv_letters"], dtype=np.float32) / 127), axis=-1)
        if self.cache and view.inventory_text is self._inventory_text:
            inventory_tokens = self._inventory_tokens
        else:
            inventory_tokens = np.stack([tokenize(s) for s in view.inventory_text])
            if self.cache:
                self._inventory_text, self._inventory_tokens = view.inventory_text, inventory_tokens
        if self.cache and view.menu_text is self._menu_text:
            menu_tokens = self._menu_tokens
        else:
            menu_tokens = np.stack([tokenize(s) for s in view.menu_text])
            if self.cache:
                self._menu_text, self._menu_tokens = view.menu_text, menu_tokens
        result = {
            "glyphs": glyphs,
            "crop": crop,
            "stats": np.concatenate((transformed, bits)).astype(np.float32),
            "message": tokenize(public_text(obs["message"])),
            "inventory": inventory_tokens,
            "inv_glyphs": inv_glyphs,
            "inv_aux": inv_aux.astype(np.float32),
            "menu": menu_tokens,
            "menu_aux": view.menu_selected,
            "mode": np.array([view.mode], dtype=np.int16),
            "previous": np.array([self.codec.previous], dtype=np.int16),
            "action_mask": view.mask,
        }
        if ui is not None:
            result.update(ui=ui, text_bytes=self._ordered_text(obs))
        if self.full_cache:
            # NLE can reuse/mutate its observation arrays in place; cached map
            # values must own their storage, unlike the ordinary one-step view.
            result["glyphs"] = result["glyphs"].copy()
            self._full_key, self._full_value = key, dict(result)
        return result
