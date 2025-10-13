import re
from typing import Dict


def remove_tags(text: str) -> str:
    """Remove XML tags from text."""
    text = re.sub(r"</?(ja|dis|unsure|song|name)>", "", text)
    
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)
        
    return text.strip()


def hiragana_to_romaji(text: str,
                       romaji_mapping: Dict[str, str]) -> str:
    """Mapping from hiragana to romaji."""
    i = 0
    result = []
    if "ヴ" in text:
        text = re.sub(r"ヴ", "ゔ", text)
    # standardize spaces
    text = re.sub("　", " ", text)
    text = re.sub(" +", " ", text)  # Remove extra spaces
    
    while i < len(text):
        match = None
        
        # Try longest match first
        for length in range(3, 0, -1):  # Max length is 3 (e.g., "っきゃ")
            if i + length <= len(text) and text[i:i+length] in romaji_mapping:
                match = text[i:i+length]
                break
                
        if match:
            result.append(romaji_mapping[match])
            i += len(match)
            
        elif text[i] == "ー" and result:  # Long vowel duplication rule
            result.append(result[-1][-1])
            i += 1
        
        else:
            result.append(text[i])  # Preserve unknown characters
            i += 1
    
    result = "".join(result)
    # add a space after punctuation symbols
    result = re.sub(r"([.,!?;:])", r"\1 ", result)  # Add space after punctuation
    result = re.sub(r"\s+", " ", result)  # Remove extra spaces
    result = result.strip()  # Split to handle multiple spaces
    
    return result


def hiragana_to_phoneme(text: str,
                        phoneme_mapping: Dict[str, str]) -> str:
    """Mapping from hiragana to phonemic representation."""
    i = 0
    result = []

    # standardize /v/
    text = re.sub(r"ヴ", "ゔ", text)
    # standardize spaces
    text = re.sub("　", " ", text)
    text = re.sub(" +", " ", text)  # Remove extra spaces

    while i < len(text):
        match = None
        
        # Try longest match first
        for length in range(3, 0, -1):  # Max length is 3 (e.g., "っきゃ")
            if i + length <= len(text) and text[i:i+length] in phoneme_mapping:
                match = text[i:i+length]
                break
                
        if match:
            result.append(phoneme_mapping[match])
            i += len(match)
        elif text[i] == "ー" and result:  # Long vowel duplication rule
            result.append(result[-1][-1])
            i += 1
        else:
            result.append(text[i])  # Preserve unknown characters
            i += 1
    
    # Post-process for "N" and "N̥" realization
    tmp_result = "".join(result)
    tmp_result = tmp_result.split()

    result = []
    for word in tmp_result:
        for j in range(len(word) - 1):
            if word[j] == "̥":
                continue
                
            if word[j] == "N" and word[j+1] == "̥":
                # bilabial
                if re.match(r"[pbm]", word[j+2]):
                    # +2 because +1 is the devoicing symbol
                    result.append("m̥")
                # velar
                elif re.match(r"[kg]", word[j+2]):
                    result.append("ŋ̥")
                # dental
                elif re.match(r"[tdn]", word[j+2]):
                    result.append("n̥")
                elif re.match(r"N", word[j+2]):
                    if j < len(word) - 3:
                        if re.match(r"[pbm]", word[j+3]):
                            result.append("m̥")
                        elif re.match(r"[kg]", word[j+3]):
                            result.append("ŋ̥")
                        elif re.match(r"[tdn]", word[j+3]):
                            result.append("n̥")
                    else:
                        result.append("ɴ̥")
                else:
                    # result[j] = "ɴ̥"  # Combining ring above (̥)
                    result.append("ɴ̥")
            elif word[j] == "N":
                # bilabial
                if re.match(r"[pbm]", word[j+1]):
                    result.append("m")
                # velar
                elif re.match(r"[kg]", word[j+1]):
                    result.append("ŋ")
                # dental
                elif re.match(r"[tdn]", word[j+1]):
                    result.append("n")
                elif word[j+1] == "N":
                    if j < len(word) - 2:
                        if re.match(r"[pbm]", word[j+2]):
                            result.append("m")
                        elif re.match(r"[kg]", word[j+2]):
                            result.append("ŋ")
                        elif re.match(r"[tdn]", word[j+2]):
                            result.append("n")
                    else:
                        result.append("ɴ")
                else:
                    result.append("ɴ")
            else:
                result.append(word[j])
        if word[-1] == "N":
            # word-final /ɴ/
            result.append("ɴ")
        else:
            result.append(word[-1])
        result.append(" ")
    
    result = "".join(result)
    # add a space after punctuation symbols
    result = re.sub(r"([.,!?;:])", r"\1 ", result)  # Add space after punctuation
    result = re.sub(r"\s+", " ", result)  # Remove extra spaces
    result = result.strip()  # Split to handle multiple spaces
    
    return result # ignore the last space


test_hiragana = [
    "まえがきばが つーふぁい っヴぁがまぬ ときゃんどぅ んまー かん ないたい。",
    "あやひどぅ ばー ぱーん そぅだてぃらいたい。",
    "ぱーや がっこうまい いでぃっだんば ずーまい ゆまいん かつまいひらいん やまとぅむぬいまい あっじゃいん ひとぅどぅあたい。",
    "ばんとぅ ぱーとぅが むぬいや いつまい すまぬ むぬいどぅ あたい。",
    "また がっこうん いきーまい すまぬ むぬいどぅ はーさかたい。",
    "あやひー っさらぬ こーこーんかい いきー やまとぅむぬいや まーぬ むぬっじゃいっだん。",
    "みどぅんあぐたが なうやひー ひろゆきゃー こーこーあいきゃ なう ちゅんま むぬっじゃだ うたいそぅが んなまー なうやひー うんそぅく むぬっじゅーがてぃ つきゅーば あがい うらー なうぬ くとぅー あらん、",
    "やまとぅ むぬいぬ あっじゃいんだきどぅ あたいてぃー あいてぃがー ばらいー うたい。",
    "だいがくー いでぃー こーこーぬ しーしーん なりー やまとぅ むぬい ほぅっでぃー  あすてぃがー ふっちゃ くぱりーうたい。",
    "すまぬ むぬいぬ じてんぬ っちゅふぁっでぃ うむいたい むぬー んなまから  17かねんまいどぅ あたい。",
    "ばー いつまい うなが まっふぁぬ あーぎん ほーげん のーとを うっきー うむいいだすが かず かきゅーたい。",
    "ゆなか すばい っさっでぃー あすとぅきゃんどぅ ゆながい つかいや みーん むぬいまい うむいだはい うたいば っふぁどぅくまんまい かきゅーたい。",
    "すばいや っしーってぃから  かかっでぃー あすてぃがー んめ きつぎんてぃー ばっしーどぅ うたい。",
    "あいえいば っふぁどぅくま やらばんまい わいてぃー かかだかーならっだん。",
    "くぬ じてんな たくぼしーしーや とーきょー、いわさきしーしーや ろすあんぜるす、ばー うつなーん うとぅい、いふてぃまい へんしゅーかいぎゅー ひーったい。",
    "あやひーまた あくせんとぬ いがらし しーしー へんしゅーぬ なかがわ しーしーが そぅいー やぐみ じゃうたう あたい。",
    "くらー  17 かねんまいん すきゃきたい むぬ えいば うなが かきゃー むぬまい ばっしゅー むぬまい はーさ あたい。",
    "あいえいば すまぬ あにそぅじゃんみんから ゆー ならいまい あすたい。",
    "いけまよしこ しーしー はずみ つむでぃまい  ほぅだ ならーす じゃうっじゃひー ふぃーさまいー やぐみ すでぃがほー。",
    "あやひーまた たくぼしーしー はずみ こくりつこくごけんきゅーしょぬ やびとぅーんーな  やぐみ すでぃがほー。",
]


test_romaji = [
    "maegakibaga cIIfai vvagamanu tokyaNdu Nmaa kaN naitai.",
    "ayahidu baa paaN sudatiraitai.",
    "paaya gakkoumai ididdaNba zIImai yumaiN kacImaihiraiN yamatumunuimai azzyaiN hituduatai.",
    "baNtu paatuga munuiya icImai sImanu munuidu atai.",
    "mata gakkouN ikiimai sImanu munuidu haasakatai.",
    "ayahii ssaranu kookooNkai ikii yamatumunuiya maanu munuzzyaiddaN.",
    "miduNagutaga nauyahii hiroyukyaa kookooaikya nau cyuNma munuzzyada utaisuga Nnamaa nauyahii uNsuku munuzzyuugati cIkyuuba agai uraa naunu kutuu araN,",
    "yamatu munuinu azzyaiNdakidu ataitii aitigaa baraii utai.",
    "daigakuu idii kookoonu siisiiN narii yamatu munui huddii asItigaa fuccya kupariiutai.",
    "sImanu munuinu ziteNnu ccyufaddi umuitai munuu Nnamakara 17kaneNmaidu atai.",
    "baa icImai unaga maffanu aagiN hoogeN nootowo ukkii umuiidasIga kazI kakyuutai.",
    "yunaka sIbai ssaddii asItukyaNdu yunagai cIkaiya miiN munuimai umuidahai utaiba ffadukumaNmai kakyuutai.",
    "sIbaiya ssiittikara kakaddii asItigaa Nme kicIgiNtii bassiidu utai.",
    "aieiba ffadukuma yarabaNmai waitii kakadakaanaraddaN.",
    "kunu ziteNna takubosiisiiya tookyoo, iwasakisiisiiya rosIaNzerusI, baa ucInaaN utui, ifutimai heNsyuukaigyuu hiittai.",
    "ayahiimata akuseNtonu igarasi siisii heNsyuunu nakagawa siisiiga suii yagumi zyautau atai.",
    "kuraa 17 kaneNmaiN sIkyakitai munu eiba unaga kakyaa munumai bassyuu munumai haasa atai.",
    "aieiba sImanu anisuzyaNmiNkara yuu naraimai asItai.",
    "ikemayosiko siisii hazImi cImudimai huda naraasI zyauzzyahii fiisamaii yagumi sIdigahoo.",
    "ayahiimata takubosiisii hazImi kokuricIkokugokeNkyuusyonu yabituuNNna yagumi sIdigahoo.",
]

test_phoneme = [
    "maegakibaga tsɨɨfai vvagamanu tokjandu mmaa kaɴ naitai.",
    "ajahidu baa paaɴ sudatiraitai.",
    "paaja gakkoumai ididdamba zɨɨmai jumaiɴ katsɨmaihiraiɴ jamatumunuimai aʑʑaiɴ hituduatai.",
    "bantu paatuga munuija itsɨmai sɨmanu munuidu atai.",
    "mata gakkouɴ ikiimai sɨmanu munuidu haasakatai.",
    "ajahii ssaranu kookooŋkai ikii jamatumunuija maanu munuʑʑaiddaɴ.",
    "miduɴagutaga naujahii hirojukjaa kookooaikja nau tɕumma munuʑʑada utaisuga nnamaa naujahii uɴsuku munuʑʑuugati tsɨkjuuba agai uraa naunu kutuu araɴ,",
    "jamatu munuinu aʑʑaindakidu ataitii aitigaa baraii utai.",
    "daigakuu idii kookoonu ɕiiɕiiɴ narii jamatu munui huddii asɨtigaa futtɕa kupariiutai.",
    "sɨmanu munuinu ʑitennu ttɕufaddi umuitai munuu nnamakara 17kanemmaidu atai.",
    "baa itsɨmai unaga maffanu aagiɴ hoogeɴ nootowo ukkii umuiidasɨga kazɨ kakjuutai.",
    "junaka sɨbai ssaddii asɨtukjandu junagai tsɨkaija miiɴ munuimai umuidahai utaiba ffadukumammai kakjuutai.",
    "sɨbaija ɕɕiittikara kakaddii asɨtigaa mme kitsɨgintii baɕɕiidu utai.",
    "aieiba ffadukuma jarabammai waitii kakadakaanaraddaɴ.",
    "kunu ʑitenna takuboɕiiɕiija tookjoo, iwasakiɕiiɕiija rosɨaɴzerusɨ, baa utsɨnaaɴ utui, ifutimai heɴɕuukaigjuu hiittai.",
    "ajahiimata akusentonu igaraɕi ɕiiɕii heɴɕuunu nakagawa ɕiiɕiiga suii jagumi ʑautau atai.",
    "kuraa 17 kanemmaiɴ sɨkjakitai munu eiba unaga kakjaa munumai baɕɕuu munumai haasa atai.",
    "aieiba sɨmanu anisuʑammiŋkara juu naraimai asɨtai.",
    "ikemajoɕiko ɕiiɕii hazɨmi tsɨmudimai huda naraasɨ ʑauʑʑahii fiisamaii jagumi sɨdigahoo.",
    "ajahiimata takuboɕiiɕii hazɨmi kokuritsɨkokugokeŋkjuuɕonu jabituunnna jagumi sɨdigahoo."
]


def test():
    """
    Test the hiragana to romaji and phoneme conversion functions.
    """
    import json
    
    # Load the mappings from JSON files
    with open('romaji_map.json', 'r', encoding='utf-8') as f:
        romaji_mapping = json.load(f)
        
    with open('phoneme_map.json', 'r', encoding='utf-8') as f:
        phoneme_mapping = json.load(f)
    
    for i, (hiragana, romaji, phoneme) in enumerate(zip(test_hiragana, test_romaji, test_phoneme)):
        assert hiragana_to_romaji(hiragana, romaji_mapping) == romaji, \
            (
                f"Test {i+1} failed for Romaji conversion from {hiragana}. "
                f"Expected: {romaji}, but got: {hiragana_to_romaji(hiragana, romaji_mapping)}. "
            )
            
        assert hiragana_to_phoneme(hiragana, phoneme_mapping) == phoneme, \
            (
                f"Test {i+1} failed for Phoneme conversion from {hiragana}. "
                f"Expected: {phoneme}, but got: {hiragana_to_phoneme(hiragana, phoneme_mapping)}. "
            )
    
    print("All tests passed!")


if __name__ == "__main__":
    test()