import re
import urllib.request
from bs4 import BeautifulSoup
from collections import defaultdict
#import whois
from urllib.parse import urlparse
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
#import ipaddress
#import socket
import requests
#from googlesearch import search
#from datetime import date, datetime
#import time
#from dateutil.parser import parse as date_parse

class FeatureExtraction:
    features = []
    def __init__(self,url):
        self.features = []
        self.df = pd.DataFrame()
        self.url = url
        self.domain = ""
        self.whois_response = ""
        self.urlparse = None
        self.response = None
        self.soup = ""
        self.d = defaultdict(LabelEncoder)
        

        try:
            self.response = requests.get(url)
            self.soup = BeautifulSoup(self.response.text, 'html.parser')
        except:
            pass

        try:
            self.urlparse = urlparse(url)
            self.domain = self.urlparse.netloc
        except:
            pass

        #try:
            #self.whois_response = whois.whois(self.domain)
        #except:
            #pass


        #URL Feature
        #self.features.append(self.getURL())
        self.features.append(self.isHttps())
        #self.features.append(self.isDomainIp())
        self.features.append(self.tld())
        self.features.append(self.URLlength())
        self.features.append(self.NoOfSubdomain())
        self.features.append(self.NoOfDots())
        self.features.append(self.NoOfObfuscatedChar())
        self.features.append(self.NoOfEqual())
        self.features.append(self.NoOfQmark())
        self.features.append(self.NoOfAmp())
        self.features.append(self.NoOfDigits())

        #HTML Feature
        self.features.append(self.LineLength())
        self.features.append(self.hasTitle())
        self.features.append(self.hasMeta())
        self.features.append(self.hasFavicon())
        self.features.append(self.hasExternalFormSubmit())
        self.features.append(self.hasCopyright())
        self.features.append(self.hasSocialNetworking())
        self.features.append(self.hasPasswordField())
        self.features.append(self.hasSubmitButton())
        self.features.append(self.hasKeywordBank())

        self.features.append(self.hasKeywordPay())
        self.features.append(self.hasKeywordCrypto())
        self.features.append(self.NoOfPopup())
        self.features.append(self.NoOfiframe())
        self.features.append(self.NoOfImage())
        self.features.append(self.NoOfJS())
        self.features.append(self.NoOfCSS())
        self.features.append(self.NoOfURLRedirect())
        self.features.append(self.NoOfHyperlink())

        #Derive Feature
        self.features.append(self.SuspiciousCharRatio())
        self.features.append(self.URLComplexityScore())
        self.features.append(self.HTMLContentDensity())
        self.features.append(self.InteractiveElementDensity())

    # 1. URL
    def getURL(self):
        url =  self.url
        return url

    # 2. isHTTPS
    def isHttps(self):
        try:
            https = self.urlparse.scheme
            if 'https' in https:
                return 1
            return 0
        except:
            return 0
            
    # 3. UsingIp
    def isDomainIp(self):
        try:
            ipaddress.ip_address(self.url)
            return 1
        except:
            return 0

    # 4. TLD
    def tld(self):
        try:
            tld = self.domain.split('.')[-1] if '.' in self.domain else ''
            #tld = self.label_encoder.fit([tld])
            return tld
        except:
            return ""

    # 5. URLlength
    def URLlength(self):
        url_length = len(self.url)
        return url_length

    # 6. NoOfSubDomain
    def NoOfSubdomain(self):
        subdomains = self.domain.split('.')[:-2] if len(self.domain.split('.')) > 2 else []
        no_of_subdomain = len(subdomains)
        return no_of_subdomain

    # 7. NoOfDots
    def NoOfDots(self):
        no_of_dots = self.url.count('.')
        return no_of_dots

    # 8. NoOfObfuscatedChar
    def NoOfObfuscatedChar(self):
        obfuscated_chars = ['%', '@', '$', '!', '+', '~', '-']
        no_of_obfuscated_chars = sum(self.url.count(char) for char in obfuscated_chars)
        return no_of_obfuscated_chars

    # 9. NoOfEqual
    def NoOfEqual(self):
        no_of_equal = self.url.count('=')
        return no_of_equal

    # 10. NoOfQmark
    def NoOfQmark(self):
        no_of_qmark = self.url.count('?')
        return no_of_qmark

    # 11. NoOfAmp
    def NoOfAmp(self):
        no_of_amp = self.url.count('&')
        return no_of_amp

    # 12. NoOfDigits
    def NoOfDigits(self):
        no_of_digits = sum(c.isdigit() for c in self.url)
        return no_of_digits
    
    # 13. LineLength
    def LineLength(self):
        html_content = self.response.text
        longest_line_length = max(len(line) for line in html_content.splitlines()) if html_content.strip() else 0
        return longest_line_length

    # 14. hasTitle
    def hasTitle(self):
        has_title = 1 if self.soup.title else 0
        return has_title

    # 15. hasMeta
    def hasMeta(self):
        try:
            return 1 if self.soup.find_all("meta") else 0
        except:
            return 0


    # 16. hasFavicon
    def hasFavicon(self):
        try:
            return 1 if self.soup.find("link", rel="icon") or self.soup.find("link", rel="shortcut icon") else 0
        except:
            return 0

    # 17. hasExternalFormSubmit
    def hasExternalFormSubmit(self):
        forms = self.soup.find_all("form")
        has_external_form_submit = 0
        for form in forms:
            action = form.get("action", "")
            if action and not action.startswith("/") and not action.startswith(self.url):
                has_external_form_submit = 1
                break
        return has_external_form_submit


    # 18. hasCopyright
    def hasCopyright(self):
        has_copyright = 1 if re.search(r"copyright", self.response.text, re.IGNORECASE) else 0
        return has_copyright

    # 19. hasSocialNetworking
    def hasSocialNetworking(self):
        social_keywords = ["facebook", "twitter", "instagram", "linkedin", "youtube"]
        has_social_networking = 1 if any(keyword in self.response.text.lower() for keyword in social_keywords) else 0
        return has_social_networking

    # 20. hasPasswordField
    def hasPasswordField(self):
        has_password_field = 1 if self.soup.find("input", {"type": "password"}) else 0
        return has_password_field
        

    # 21. hasSubmitButton
    def hasSubmitButton(self):
        has_submit_button = 1 if self.soup.find("button", {"type": "submit"}) or self.soup.find("input", {"type": "submit"}) else 0
        return has_submit_button

    # 22. hasKeywordBank
    def hasKeywordBank(self):
        has_keyword_bank = 1 if re.search(r"\bbank\b", self.response.text, re.IGNORECASE) else 0
        return has_keyword_bank
    
    # 23. hasKeywordPay
    def hasKeywordPay(self):
        has_keyword_pay = 1 if re.search(r"\bpay\b", self.response.text, re.IGNORECASE) else 0
        return has_keyword_pay

    # 24. hasKeywordCrypto
    def hasKeywordCrypto(self):
        has_keyword_crypto = 1 if re.search(r"\bcrypto\b", self.response.text, re.IGNORECASE) else 0
        return has_keyword_crypto

    # 25.  NoOfPopup
    def NoOfPopup(self):
        no_of_popup = self.response.text.lower().count("popup")
        return no_of_popup

    # 26.  NoOfiFrame
    def NoOfiframe(self):
        no_of_iframe = len(self.soup.find_all("iframe"))
        return no_of_iframe 

    # 27.  NoOfImage
    def NoOfImage(self):
        no_of_image = len(self.soup.find_all("img"))
        return no_of_image 

    # 28.  NoOfJS
    def NoOfJS(self):
        no_of_js = len(self.soup.find_all("script"))
        return  no_of_js  

    # 29.  NoOfCSS
    def NoOfCSS(self):
        no_of_css = len(self.soup.find_all("link", rel="stylesheet"))
        return  no_of_css 
        
    # 30.  NoOfURLRedirect
    def NoOfURLRedirect(self):
        redirects = self.response.history
        no_of_url_redirect = len(redirects)
        return  no_of_url_redirect 
        
    # 31. NoOfHyperlink
    def NoOfHyperlink(self):
        no_of_hyperlink = len(self.soup.find_all("a"))
        return  no_of_hyperlink
        
    # 32. Derived Feature: SuspiciousCharRatio
    def SuspiciousCharRatio(self):
        suspicious_char_ratio = (
                    self.NoOfObfuscatedChar() +
                    self.NoOfEqual() +
                    self.NoOfQmark() +
                    self.NoOfAmp()
                ) / self.URLlength()
        return suspicious_char_ratio

    # 33. Derived Feature: SuspiciousCharRatio
    def URLComplexityScore(self):
        first_term = (
            self.URLlength() +
            self.NoOfSubdomain() +
            self.NoOfObfuscatedChar()
        ) / self.URLlength()
    
        second_term = (
            self.NoOfEqual() +
            self.NoOfAmp()
        ) / (self.NoOfQmark() + 1)
    
        # Calculate the URL Complexity Score
        url_complexity_score = first_term + second_term
        return url_complexity_score

    # 34. Derived Feature: HTMLContentDensity
    def HTMLContentDensity(self):
        # Calculate the HTML Content Density
        html_content_density = (
            self.LineLength() + self.NoOfImage()
        ) / (
            self.NoOfJS() + self.NoOfCSS() + self.NoOfiframe() + 1  # Add 1 to avoid division by zero
        )
        return html_content_density

    # 35. InteractiveElementDensity
    def InteractiveElementDensity(self):
        # Calculate the Interactive Element Density
        interactive_element_density = (
            self.hasSubmitButton() +
            self.hasPasswordField() +
            self.NoOfPopup()
        ) / (
            self.LineLength() + self.NoOfImage()
        )
        return interactive_element_density

    def getFeaturesList(self):
        #df = pd.DataFrame(self.features)
       # d = defaultdict(LabelEncoder)
        #df = df.apply(lambda x: d[x.name].fit_transform(x))
        #self.features = self.features.apply(lambda x: d[x.name].fit_transform(x))
        #data = pd.DataFrame(self.features)
        # Apply LabelEncoder to each column and collect features
        #encoded_data = data.apply(lambda col: d[col.name].fit_transform(col))
        #self.features = encoded_data.values.tolist()
        #print("getFeaturesList ok")
        return self.features

    def getLabelEncoder(self):
        #df = pd.DataFrame(self.features)
        obj = np.array(self.getFeaturesList()).reshape(1,33) 
        df = pd.DataFrame(obj)
        d = defaultdict(LabelEncoder)
        #df = df.apply(lambda x: d[x.name].fit_transform(x))
        encoded_df = df.apply(lambda col: d[col.name].fit_transform(col))
        #data = pd.DataFrame(self.features)
        # Apply LabelEncoder to each column and collect features
        #encoded_data = data.apply(lambda col: d[col.name].fit_transform(col))
        #self.features = encoded_data.values.tolist()
        #print("getFeaturesList ok")
        
        return  encoded_df
