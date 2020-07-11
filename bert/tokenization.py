# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import unicodedata
import six
import tensorflow as tf


def validate_case_matches_checkpoint(do_lower_case, init_checkpoint):
  """Checks whether the casing config is consistent with the checkpoint name."""

  # The casing has to be passed in by the user and there is no explicit check
  # as to whether it matches the checkpoint. The casing information probably
  # should have been stored in the bert_config.json file, but it's not, so
  # we have to heuristically detect it to validate.

  if not init_checkpoint:
    return

  m = re.match("^.*?([A-Za-z0-9_-]+)/bert_model.ckpt", init_checkpoint)
  if m is None:
    return

  model_name = m.group(1)

  lower_models = [
      "uncased_L-24_H-1024_A-16", "uncased_L-12_H-768_A-12",
      "multilingual_L-12_H-768_A-12", "chinese_L-12_H-768_A-12"
  ]

  cased_models = [
      "cased_L-12_H-768_A-12", "cased_L-24_H-1024_A-16",
      "multi_cased_L-12_H-768_A-12"
  ]

  is_bad_config = False
  if model_name in lower_models and not do_lower_case:
    is_bad_config = True
    actual_flag = "False"
    case_name = "lowercased"
    opposite_flag = "True"

  if model_name in cased_models and do_lower_case:
    is_bad_config = True
    actual_flag = "True"
    case_name = "cased"
    opposite_flag = "False"

  if is_bad_config:
    raise ValueError(
        "You passed in `--do_lower_case=%s` with `--init_checkpoint=%s`. "
        "However, `%s` seems to be a %s model, so you "
        "should pass in `--do_lower_case=%s` so that the fine-tuning matches "
        "how the model was pre-training. If this error is wrong, please "
        "just comment out this check." % (actual_flag, init_checkpoint,
                                          model_name, case_name, opposite_flag))


def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
  """Returns text encoded in a way suitable for print or `tf.logging`."""

  # These functions want `str` for both Python2 and Python3, but in one case
  # it's a Unicode string and in the other it's a byte string.
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text
    elif isinstance(text, unicode):
      return text.encode("utf-8")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
  """Loads a vocabulary file into a dictionary."""
  vocab = collections.OrderedDict()
  index = 0
  with tf.gfile.GFile(vocab_file, "r") as reader:
    while True:
      token = convert_to_unicode(reader.readline())
      if not token:
        break
      token = token.strip()
      vocab[token] = index
      index += 1
    print(len(vocab))
    # letter_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J', 'K', 'L', 'M', 'N',
    #                'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '覈', '/', '-', '(', ')','%', '【', '丨', '】',
    #                '忔', 'ǎ', '\ue035', 'Ｉ', '痦', '彳', '溦', '阄', '’', '\ue310', '▊', '亸', 'Ｅ', 'Ａ', '\ue220', '睍', '旂',
    #                'é', '嚯', '┇', '槊', '茕', '劼', '涜', '鍙', '亻', '龉', '蕞', 'ā', '旳', '⒏', '哃', '瘜', '殄', '帀', 'Ⅱ',
    #                '\ue222', '`', 'Ｒ', '蝮', '镚', '郗', '姇', '鐢', '泺', '卝', '烎', '疄', '∫', '瑧', '‐', '叒', '閮', '\ue21e',
    #                '\ue41f', '亓', '桭', '―', '罝', 'Ｄ', '毖', '跬', '∨', '屼', '\ue13d', '鎸', '梿', '莜', '赟', '\ue231', '亍',
    #                '曁', '笫', '瀵', '⒎', '…', 'ē', '椴', 'Ｆ', '呭', '\ue21d', '\ue317', '\ue221', '犇', '噃', '飡', '聃', '叕',
    #                '殚', '‘', '靊', '锛', '戝', '\ue253', '歀', 'Ι', '佺', 'è', '\ue32f', '缁', '弢', 'Ｎ', '氺', '轫', '岿',
    #                '\ue40a', '辔', '镝', '\ue345', '呺', '鐜', '鍒', '⒐', '茆', '綉', '勮', '\ue21c', '莳', '\ue627', '鑱', '笪',
    #                '嗮', '祡', '\ue21f', '⒌', '\ue66d', '\ue11d', '⒍', '↙', '铗', '龃', '–', 'ú', '㈠', '讠', '郯', '祹', '簰',
    #                '尓', '蚨', '”', '雬', '“', '—', '鑷', '鏉',
    #
    #                '🐝', '➍', '🇼', '🏻', '屃', '易', '❷', '👼', '🥂', '۶', 'Ｋ', '浥', '️', '💛', '📢', '🙌', '💗', '❎',
    #                '♐', '💝', '❣', '📊', '𝙁', '荑', '✳', '🎁', '︎', '愠', '⚡', '犴', '🆘', '🔝', '髫', '谡', '𝙚', '🛍',
    #                '🔚', '搴', '鲮', '庒', '\ue04f', 'Á', '捭', '💱', '🈷', '撺', '🇵', '戋', '🈵', 'Ｓ', '椱', '🌻', '⚠', '🎗',
    #                '歃', '♒', '👇', '浐', '汯', '❻', '🤝', '🎏', '❇', 'ℑ', '🇻', '⽐', '👊', '⾦', '畺', 'Φ', '𝐀', '𝐌', '昰',
    #                '嗌', '秕', '♨', '🇦', '褴', '翀', '髄', '🌹', '❹', '✌', '🏽', '🏦', '⁉', '崃', '🈶', '🍒', '🌲', '➢', '✧',
    #                '😘', '垕', 'Ｕ', '汭', '🌴', '暍', '嶶', '🌇', '🔑', '🍎', '⛏', '劖', '🏅', '荭', '魉', '🏇', 'Ⅵ', '㊗', '諨',
    #                '𝙧', '锎', '🎀', '🔆', '阓', '🙅', '\ue312', '杲', 'Ⅰ', '⾼', '👑', '谝', '曌', '緈', '⛎', '🌅', '💯',
    #                '\ue18d', '🇪', '\ue796', '🐔', '하', '♻', 'Ë', '🇸', '🙃', '⼀', '⑬', 'ł', '曩', '皁', 'Ł', '📉',
    #                '\ue779', '🐱', '🔓', '媖', '🇫', '畾', '💎', 'Ｍ', '咑', '🍰', '🇲', '㊊', '🤗', '📌', '婂', '₳',
    #                '\ue12f', '扌', '🌊', '👆', '鄠', '💚', '🌸', '⽅', '韫', '搛', '釆', '🇬', '⑪', '🐠', '⓿', '堞', '囵', '🇨',
    #                '🙏', '鎵', 'ⅹ', '🇰', '💹', '🌳', '鸃', '毑', '∷', '\ue41d', 'Ｘ', '鋆', '٩', '👻', '彽', '🏩', '溆', '❽',
    #                '🇷', '蚰', '🎋', '✊', '塱', '狲', '‼', 'ê', '😝', '오', '💬', '﹌', '𝐄', '🌚', '鸱', '螭', '🌍', 'Ｐ', '铖',
    #                '⼤', '😱', '🇾', '😀', '🛡', '訾', '\ue021', '😉', '侪', '◀', '🇱', '🔱', '𝐒', '🐮', '圪', '诨', '💪',
    #                '❶', '민', '浡', '😍', '🇹', '🌏', '🌱', '鹆', '㊙', '✒', '⏰', '笧', '💥', '𝐍', '☎', '猊', '🔅', '🔒',
    #                '🔛', '🉐', '锞', '邡', '💡', '⬇', '渽', '🚩', '魑', '💞', '\ue04c', '\ue307', '鸩', '🇮', '🐻', '🌺',
    #                '揺', '抃', '￼', '💫', '\U0001f9e1', '骝', '🐷', '菥', '💲', '捯', '𝐓', '﹀', '蓥', '🔰', '🌙', 'Ｔ', '阛',
    #                '亯', '▆', '邴', '斲', '逋', 'Ｏ', '😻', '🏧', '➕', '🐂', '💙', '🎈', '☑', '영', '锃', '🚀', '🕳', '幤',
    #                '🏃', '铧', '祼', '🇺', '🌷', '➊', '跶', '☟', '❗', '⑫', '〰', '穏', '鲳', '❈', '睚', '❔', '\uf06c', '박',
    #                '栢', '骟', '😁', '👈', '𝙯', '鐿', '槗', '狻', '준', '⾃', '✎', '𝙖', '👉', '🐽', '¢', '나', '🍀', '\ue107',
    #                '😌', '\ue112', '✤', '\ue14c', '😡', '💐', '🚢', '🇯', '阍', '☝', '姤', '🏵', '💵', '┋', '℡', '🔜',
    #                '玎', '✘', '\ue6fd', '🏆', '❼', '丅', '桄', '❾', '👏', '檩', '啭', '𝐔', 'ご', 'ﻪ', '燚', '😊', '\ue622',
    #                '狥', '瘕', '➎', '钅', '藁', '娭', '🇩', '⭐', '❓', '➋', '狴', '🌰', '➡', '⇙', '💉', '🇳', '🍞', '廋', '⃣',
    #                '♏', '🇧', '💍', '瓞', '≠', '❸', '😏', '嗾', '芘', '⽤', '🌼', '🔟', '⽣', '舢', '▏', '🔍', '💰', '➌', '惢',
    #                '黉', '🌟', '🔴', '猢', '📈', '🍇', '🎊', '𝙪', '𝙩', '♉', '➗', '⽇', '茀', '圑', '閞', '牖', '寳', '鲅',
    #                '🇽', '銆', '綯', '🎉', '🕹', '🚄', '❺', '慱', '⼈', '楒', 'Ｃ', 'Ｇ', '🌎', '▣', '☜', '💴', '\ue246', '𝘽',
    #                '🌈', '冚', '🇴', '🇭'
    #
    #                '\ue030', '\ue332', '熻', '\ue022', '\ue60a', '\ue00e', '\ue608', '犟', '\ue131', '\ue60e', '\ue609',
    #                '┗', '诓', '\ue219', '\ue607', '蚬', '┓', '\ue40b', '\ue10d', '讣', '\ue612', '\ue110', '缃', '鍑',
    #                '\ue611', '\ue60b', '\ue114', '\ue00d', '貹', '┄', '\ue60f', '┏', '\ue60c', '戣', '嚚', '┛',
    #
    #                '④', '※', '諮', '$', 'ｏ', '９', '^', '●', '員', '▲', '７', '爲', '險', 'ｐ', '─', '園', '㎡', '貨', '│', '亞',
    #                '營', '東', '術', '報', '預', '⒋', '△', '萬', '們', '廣', '瀏', '*', '徬', '複', '風', 'の', '匯', 'ａ', '①', '計',
    #                '馮', '≥', '√', '隊', 'ー', '５', '８', 'ｖ', '+', '樂', '內', '圖', '⑨', '児', '載', '責', '碼', '絶', '怼', '萊',
    #                '冏', '無', '☆', '蓋', '∞', '題', '錯', '「', '{', '‖', '％', '見', '⑥', '詐', '幣', '︱', '鬆', '啟', '隨', '○',
    #                '經', '鑽', '娛', '權', '訊', '孖', '現', '趨', '問', '過', '區', '遠', '構', '║', '僱', '〗', '→', '產', '．', '覽',
    #                '０', '⒈', '呲', '鏈', '⑧', '』', '↓', '囍', '誰', '絡', '機', '資', '庫', '為', '給', '６', '〞', '♂', '設', '業',
    #                '標', '－', '間', '變', '〕', '長', '&', '實', '這', '談', '◇', '≧', '創', '橫', '塗', "'", '〝', '∣', '連', '總',
    #                'ｄ', '賠', '垚', '謀', '≈', '⑩', '將', '~', '簡', 'θ', '關', '會', '倉', '決', ']', '／', '〈', '戰', '勢', '塊',
    #                '氫', '進', '價', '負', '潤', '時', '⒉', '漢', '擔', 'ｎ', '獨', '繫', '_', '傳', '郵', '沒', '▍', '華', '↗', '１',
    #                '擁', '鮮', '騙', '『', '◆', '離', '級', '□', '＞', '視', '黃', '專', '電', '捌', '場', '際', '數', '雲', '■', '豐',
    #                '★', '質', '±', '務', '↑', '·', '參', '」', '戀', '薅', '〉', '鉅', '"', '４', '環', '該', '開', '▌', '▼', '〖',
    #                '斷', '=', '議', '賺', '來', '\\', '③', '⒊', '償', '艹', '興', '詢', '馬', '網', '＝', 'ｒ', '>', 'ｇ', '÷', '稅',
    #                '〔', '詳', '帳', '≡', '優', '＋', '測', '②', '儲', '聯', '艙', 'ｙ', '￥', '滿', '＄', '﹏', '鍵', '▋', '廠', '財',
    #                '｜', '國', '錢', '恆', 'ｓ', '戶', '<', '３', '費', '≤', '‰', '緊', 'ｉ', '豊', '頂', '體', '°', '～', '慮', '領',
    #                '█', '＂', '續', '龍', '還', '轉', '請', '↘', '←', '餘', '}', '×', '發', '審', '|', '規', '紅', '圏', '℃', '２',
    #                '個', '@', '強', '⑤', '⊙', '團', '⑦',
    #
    #                '穩', '與', '層', '丿', '運', '動', 'π', '達', '＜', '満', '帶', '━', '樓', '對', '購', '′', '寫', '類', '線', '沖',
    #                '點', '針', '贏', '灬', '▽', '從', '項',
    #
    #                '©', '歩', 'ｕ', '驟', '◎', '♀', '忪', '獸', '臺', 'ˇ', '製', '別', '陸', '妳', '☀', '㊣', '慶', '〜', '俬', '掛',
    #                '╰', '✕', '躍', '輝', '┊', 'ｅ', '調', '•', '惡', '譽', '梟', 'ア', 'オ', '┃', '〇', '終', '➤', '輩', '損', '協',
    #                '積', '®', '遞', '､', '裡', '▶', '◤', '靈', '銀', '๑', '丟', '講', '頻', '滙', '✖', '貳', 'ユ', '佔', '″', '寶',
    #                '»', '™', '減', '＇', '◢', '瑪', '讓', '祂', '😂', '驗', '€', '輔', '✦', '¥', '﹐', '紡', '兒', '﹑', 'ァ', '煙',
    #                '▉', '証', '｝', '結', '✪', '｛', 'β', '｡', '［', '莊', '齁', '♬', '競', '億', '＆', 'ゆ', '遊', '﹡', '當', 'ｈ',
    #                '轄', 'ｗ', '❤', 'ウ', '朮', '處', '氣', '臨', '啲', '牠', '啓', '錦', '衛', '▪', '採', 'ゅ', '恵', '♥', '淨', '禦',
    #                '▃', '願', '衝', '勝', '☞', '覺', '註', '］', '啫', '蘭', '戲', '❀', 'せ', '▂', '陳', '墮', '潰', '✨', '«', '濟',
    #                '磡', '✔', '＊', '沬',
    #
    #
    #                ]
    # for letter in letter_list:
    #   vocab[letter] = index
    #   index += 1
  return vocab


def convert_by_vocab(vocab, items):
  """Converts a sequence of [tokens|ids] using the vocab."""
  output = []
  for item in items:
    output.append(vocab[item])
  return output


def convert_tokens_to_ids(vocab, tokens):
  return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
  return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
  """Runs basic whitespace cleaning and splitting on a piece of text."""
  text = text.strip()
  if not text:
    return []
  tokens = text.split()
  return tokens


class FullTokenizer(object):
  """Runs end-to-end tokenziation."""

  def __init__(self, vocab_file, do_lower_case=True):
    self.vocab = load_vocab(vocab_file)
    self.inv_vocab = {v: k for k, v in self.vocab.items()}
    self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

  def tokenize(self, text):
    split_tokens = []
    for token in self.basic_tokenizer.tokenize(text):
      for sub_token in self.wordpiece_tokenizer.tokenize(token):
        split_tokens.append(sub_token)

    return split_tokens

  def convert_tokens_to_ids(self, tokens):
    return convert_by_vocab(self.vocab, tokens)

  def convert_ids_to_tokens(self, ids):
    return convert_by_vocab(self.inv_vocab, ids)


class BasicTokenizer(object):
  """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

  def __init__(self, do_lower_case=True):
    """Constructs a BasicTokenizer.

    Args:
      do_lower_case: Whether to lower case the input.
    """
    self.do_lower_case = do_lower_case

  def tokenize(self, text):
    """Tokenizes a piece of text."""
    text = convert_to_unicode(text)
    text = self._clean_text(text)

    # This was added on November 1st, 2018 for the multilingual and Chinese
    # models. This is also applied to the English models now, but it doesn't
    # matter since the English models were not trained on any Chinese data
    # and generally don't have any Chinese data in them (there are Chinese
    # characters in the vocabulary because Wikipedia does have some Chinese
    # words in the English Wikipedia.).
    text = self._tokenize_chinese_chars(text)

    orig_tokens = whitespace_tokenize(text)
    split_tokens = []
    for token in orig_tokens:
      if self.do_lower_case:
        token = token.lower()
        token = self._run_strip_accents(token)
      split_tokens.extend(self._run_split_on_punc(token))

    output_tokens = whitespace_tokenize(" ".join(split_tokens))
    return output_tokens

  def _run_strip_accents(self, text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
      cat = unicodedata.category(char)
      if cat == "Mn":
        continue
      output.append(char)
    return "".join(output)

  def _run_split_on_punc(self, text):
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
      char = chars[i]
      if _is_punctuation(char):
        output.append([char])
        start_new_word = True
      else:
        if start_new_word:
          output.append([])
        start_new_word = False
        output[-1].append(char)
      i += 1

    return ["".join(x) for x in output]

  def _tokenize_chinese_chars(self, text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
      cp = ord(char)
      if self._is_chinese_char(cp):
        output.append(" ")
        output.append(char)
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)

  def _is_chinese_char(self, cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
      return True

    return False

  def _clean_text(self, text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
      cp = ord(char)
      if cp == 0 or cp == 0xfffd or _is_control(char):
        continue
      if _is_whitespace(char):
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)


class WordpieceTokenizer(object):
  """Runs WordPiece tokenziation."""

  def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
    self.vocab = vocab
    self.unk_token = unk_token
    self.max_input_chars_per_word = max_input_chars_per_word

  def tokenize(self, text):
    """Tokenizes a piece of text into its word pieces.

    This uses a greedy longest-match-first algorithm to perform tokenization
    using the given vocabulary.

    For example:
      input = "unaffable"
      output = ["un", "##aff", "##able"]

    Args:
      text: A single token or whitespace separated tokens. This should have
        already been passed through `BasicTokenizer.

    Returns:
      A list of wordpiece tokens.
    """

    text = convert_to_unicode(text)

    output_tokens = []
    for token in whitespace_tokenize(text):
      chars = list(token)
      if len(chars) > self.max_input_chars_per_word:
        output_tokens.append(self.unk_token)
        continue

      is_bad = False
      start = 0
      sub_tokens = []
      while start < len(chars):
        end = len(chars)
        cur_substr = None
        while start < end:
          substr = "".join(chars[start:end])
          if start > 0:
            substr = "##" + substr
          if substr in self.vocab:
            cur_substr = substr
            break
          end -= 1
        if cur_substr is None:
          is_bad = True
          break
        sub_tokens.append(cur_substr)
        start = end

      if is_bad:
        output_tokens.append(self.unk_token)
      else:
        output_tokens.extend(sub_tokens)
    return output_tokens


def _is_whitespace(char):
  """Checks whether `chars` is a whitespace character."""
  # \t, \n, and \r are technically contorl characters but we treat them
  # as whitespace since they are generally considered as such.
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  cat = unicodedata.category(char)
  if cat == "Zs":
    return True
  return False


def _is_control(char):
  """Checks whether `chars` is a control character."""
  # These are technically control characters but we count them as whitespace
  # characters.
  if char == "\t" or char == "\n" or char == "\r":
    return False
  cat = unicodedata.category(char)
  if cat in ("Cc", "Cf"):
    return True
  return False


def _is_punctuation(char):
  """Checks whether `chars` is a punctuation character."""
  cp = ord(char)
  # We treat all non-letter/number ASCII as punctuation.
  # Characters such as "^", "$", and "`" are not in the Unicode
  # Punctuation class but we treat them as punctuation anyways, for
  # consistency.
  if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
      (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
    return True
  cat = unicodedata.category(char)
  if cat.startswith("P"):
    return True
  return False
