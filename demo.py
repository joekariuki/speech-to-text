from ibm_watson import SpeechToTextV1, LanguageTranslatorV3
import json
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from pandas import json_normalize


url_s2t = '[SPEECH TO TEXT SERVICE URL]'
url_lt = '[LANGUAGE TRANSLATOR SERVICE URL]'
iam_apikey_s2t = '[SPEECH TO TEXT API KEY]'
apikey_lt = '[LANGUAGE TRANSLATOR API KEY]'

version_lt = '2018-05-01'

speech_authenticator = IAMAuthenticator(iam_apikey_s2t)
s2t = SpeechToTextV1(authenticator=speech_authenticator)
s2t.set_service_url(url_s2t)

language_autheticator = IAMAuthenticator(apikey_lt)
language_translator = LanguageTranslatorV3(
    version=version_lt, authenticator=language_autheticator)
language_translator.set_service_url(url_lt)


filename = 'PolynomialRegressionandPipelines.mp3'

with open(filename, mode="rb") as wav:
    response = s2t.recognize(audio=wav, content_type='audio/mp3')

json_normalize(response.result['results'], "alternatives")

recognized_text = response.result['results'][0]["alternatives"][0]["transcript"]

json_normalize(
    language_translator.list_identifiable_languages().get_result(), "languages")

# Spanish Translation of Text
translation_response = language_translator.translate(
    text=recognized_text, model_id='en-es')
translation = translation_response.get_result()
spanish_translation = translation['translations'][0]['translation']
print(f"Spanish Translation: {spanish_translation}")

# Spanish to English Translation
translation_new = language_translator.translate(
    text=spanish_translation, model_id='es-en').get_result()
translation_eng = translation_new['translations'][0]['translation']
print(f"English translation: {translation_eng}")

# English to French Translation
french_translation = language_translator.translate(
    text=translation_eng, model_id='en-fr').get_result()
translation_fr = french_translation['translations'][0]['translation']
print(f"French Translation: {translation_fr}")
