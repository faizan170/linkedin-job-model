import re
import json
import os
import pandas as pd
from random import shuffle
import numpy as np

class Preprocess():
    def __init__(self):
        self.skillsAll = []
        self.skillLabel = []

    def addSkill(self, skill):
        self.skillsAll.append(skill)

    def skillWithLabel(self, skillStr, label):
        self.skillLabel.append([skillStr, label])

    def listToString(self, listData, sep = ", "):
        val = ""
        for item in listData:
            val += item + sep
        if sep == ", ":
            val = val[:-2]
        elif sep == " ":
            val = val.strip()
        return val

    def formatString(self, strData):
        new = re.sub('[^A-Za-z0-9]+', ' ', strData)
        return new.strip()

    def processSkills(self, skillsList, personProfile):
        #print(personProfile)
        allSkills = []

        if "mainSkills" in skillsList.keys():
            for d in skillsList["mainSkills"]:
                allSkills.append(d["title"])
                self.addSkill(d["title"])
        if "otherSkills" in skillsList.keys():
            for typeS in skillsList["otherSkills"]:
                for value in typeS["skills"]:
                    allSkills.append(value)
                    self.addSkill(value)

        if len(allSkills) != 0:
            allSkills = self.listToString(allSkills)
            self.skillLabel.append([allSkills, personProfile])
        else:
            allSkills = ""
        return allSkills

    def processEducation(self, educationList):
        finalStr = ""
        for education in educationList:
            if "title" in education.keys():
                finalStr += education["title"]
            if "studyArea" in education.keys():
                finalStr += " in major subject " + education["studyArea"]
            finalStr += ". "
        return finalStr.strip()

    def processCertifications(self, certifications):
        finalStr = ""
        for cert in certifications:
            if "issuingCompany" not in cert.keys():
                cert["issuingCompany"] = ''
            finalStr += "{title} certification from {issuingCompany}. ".format(**cert)
        return finalStr.strip()

    def processExperience(self, experience):
        finalStr = ""
        for exp in experience:
            if "role" in exp.keys():
                if "location" not in exp.keys():
                    exp["location"] = ""
                if "duration" not in exp.keys():
                    exp["duration"] = ""
                finalStr += "{role} in {company} for {duration} located in {location}. ".format(**exp)
                if "jobDetails" in exp.keys():
                    finalStr += exp["jobDetails"] + " "
        return finalStr.strip()

    def processAccomplishments(self, data, acType, dataStr):
        if data[acType] != None:
            for val in data[acType]:
                dataStr += val + ", "
            dataStr = dataStr[:-2]
        else:
            dataStr = ""
        return dataStr


    def getDataFrame(self, dataDir):
        finalData = []
        finalLabels = []
        print("[INFO] Processing {} records".format(len(os.listdir(dataDir))))
        for fileName in os.listdir(dataDir):
            with open(os.path.join(dataDir, fileName), "r") as file:
                singleRes = []
                data = json.load(file)
                if "profileTitle" in data.keys():
                    singleRes.append(data["profileTitle"])
                if "loc" in data.keys():
                    singleRes.append(data["loc"])
                if "about" in data.keys():
                    singleRes.append(data["about"])
                if "experience" in data.keys():
                    singleRes.append(self.processExperience(data["experience"]))
                if "education" in data.keys():
                    singleRes.append(self.processEducation(data["education"]))
                if "certifications" in data.keys():
                    singleRes.append(self.processCertifications(data["certifications"]))
                if "skills" in data.keys():
                    singleRes.append(self.processSkills(data["skills"], fileName.replace(".json","")))
                if "accomplishments" in data.keys():
                    singleRes.append(self.processAccomplishments(data["accomplishments"], "courses", "Learned following courses: "))
                    singleRes.append(self.processAccomplishments(data["accomplishments"], "projects", "Worked on following projects: "))
                    langs = self.processAccomplishments(data["accomplishments"], "languages", "Knows following languages: ")
                    singleRes.append(langs)
                    
                finalLabels.append(fileName.replace(".json",""))
                finalData.append(singleRes)
        finalDataSet = []
        for i, p in enumerate(finalData):
            label = finalLabels[i]
            value = p.copy()
            for i in range(50):
                shuffle(value)
                finalDataSet.append((self.formatString(self.listToString(value)), label))
        
        data = [self.skillsAll, self.skillLabel]
        data = np.array(data)
        df = pd.DataFrame(finalDataSet, columns=["text", "label"])
        return df
        