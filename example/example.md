## Example1

### 1) Input
- **Question **: Can the wild type *Yarrowia lipolytica* naturally consume xylose without metabolic engineering?

### 2) Retrieved Evidence
- **Doc1**: Moreover, the addition of methyl viologen as an artificial electron carrier shifted the metabolic flux from acid production to alcohol production, resulting in a high biofuel yield of 0.39 g/g from cellulose, comparable to ethanol yield from corn dextrose by yeast fermentation. This study is the first metabolic engineering of C. cellulovorans for n-butanol and ethanol production directly from cellulose with significant titers and [...] Production of cellulosic biofuels has drawn increasing attention. However, currently no microorganism can produce biofuels, particularly butanol, directly from cellulosic biomass efficiently. Here we engineered a cellulolytic bacterium, Clostridium cellulovorans, for n-butanol and ethanol production directly from cellulose by introducing an aldehyde/alcohol dehydrogenase (adhE2), which converts butyryl-CoA to n-butanol and acetyl-CoA to ethanol. The engineered strain was able to produce 1.42 [...] Keywords:  Biofuel; Butanol; Cellulose; Clostridium cellulovorans; Ethanol;
- **Doc2**: With high cellulolytic and acetic/butyric acids production abilities, Clostridium cellulovorans is promising for use to produce cellulosic n-butanol. Here, we introduced three different aldehyde/alcohol dehydrogenases encoded by bdhB, adhE1, and adhE2 from Clostridium acetobutylicum into C. cellulovorans and studied their effects on ethanol and n-butanol production. Compared to AdhE2, AdhE1 was more specific for n-butanol biosynthesis over ethanol. Co-expressing adhE1 with bdhB produced a [...] comparable amount of butanol but significantly less ethanol, leading to a high butanol/ethanol ratio of 7.0 and 5.6 (g/g) in glucose and cellulose fermentation, respectively. Co-expressing adhE1 or adhE2 with bdhB did not increase butanol production because the activity of BdhB was limited by the NADPH availability in C. cellulovorans. Overall, the strain overexpressing adhE2 alone produced the most n-butanol (4.0 g/L, yield: 0.22 ± 0.01 g/g). Based on the insights from this study, further [...] metabolic engineering of C. cellulovorans for cellulosic n-butanol production is suggested.
- **Doc3**: one of the most promising liquid biofuels for the future. Recent studies aimed at engineering n-butanol production in C. cellulovorans represent milestones towards production of biofuels through one-step fermentation of lignocellulose but also indicated that more detailed understanding of the C. cellulovorans central carbon metabolism is essential to refine metabolic engineering strategies towards improved n-butanol production in this strain. The present study [...] Clostridium cellulovorans is among the most promising candidates for consolidated bioprocessing (CBP) of cellulosic biomass to liquid biofuels (ethanol, butanol). C. cellulovorans metabolizes all the main plant polysaccharides and mainly produces butyrate. Since most butyrate and butanol biosynthetic reactions from acetyl-CoA are common, introduction of single heterologous alcohol/aldehyde dehydrogenase can divert the branching-point intermediate (butyryl-CoA) towards butanol production in this [...] related to overexpression and secretion of (hemi-)cellulases), which induces up-regulation of ATP synthetic pathways, e.g. acetate production and ATP synthase. SIGNIFICANCE: C. cellulovorans can metabolize all the main plant polysaccharides (cellulose, hemicelluloses and pectins) and, unlike other well established cellulolytic microorganisms, can produce butyrate. C. cellulovorans is therefore among the most attractive candidates for direct fermentation of lignocellulose to high-value chemicals

### 3) Question Understanding 
- **Intent**: Determine whether wild-type *Y. lipolytica* can consume xylose **naturally**.
- **Constraint**: *without metabolic engineering* (i.e., no heterologous pathways, no key-enzyme overexpression, no ALE as an enabling step).
- **Question type**: **Binary (Yes/No)**.

### 4) Prompt Template Selection
- **Prompt Pool (illustrative)**:
  - **T1**: Binary QA (Yes/No + evidence-based justification)
  - **T2**: Multiple-choice QA (option alignment + elimination)

- **Selected Template**: **T1 — Binary QA**
- **Rationale**: The question asks for a direct yes/no judgment under a strict “without engineering” constraint; mechanism-level detail is optional.

### 5) Final Prompt 
```text
[System]
You are a domain expert in yeast metabolism and strain engineering. You must answer strictly based on the provided evidence.
Return a concise Yes/No answer, then justify it with 2–4 sentences.
If the evidence indicates the wild type cannot do it, answer "No".
Cite supporting evidence using the format (Doc1), (Doc2), (Doc3).

[User]
Question:
Can the wild type Yarrowia lipolytica naturally consume xylose without metabolic engineering?

Evidence:
(Doc1) "Yarrowia lipolytica cannot naturally metabolise xylose ... heterologous xylose metabolic pathways must be engineered ... to facilitate its consumption."
(Doc2) "However, it is unable to consume xylose ... Here, we engineered Y. lipolytica to metabolize xylose ..."
(Doc3) "its lack of an active xylose metabolism restricts ... metabolic engineering and adaptive laboratory evolution were applied ... resulting in an efficient xylose-assimilating strain."

Please answer Yes/No and provide an evidence-based explanation with citations.
```

### 6) Model Output 

Answer: No.

Explanation: The evidence consistently indicates that wild-type Yarrowia lipolytica does not naturally metabolize or consume xylose. Multiple sources explicitly state it is unable to consume xylose and lacks an active xylose metabolism. Therefore, xylose utilization requires metabolic engineering (e.g., introducing heterologous xylose pathways and/or overexpressing key enzymes), and in some cases further improvement via adaptive laboratory evolution. (Doc1) (Doc2) (Doc3)



## Example 2
### 1) Input
- **Question (Q3)**: Which enzyme activity was knocked out in Synechococcus elongatus PCC 7942 to prevent free fatty acid recycling? A:Thioesterase B:Rubisco C:ATP synthase D:Acyl-ACP synthetase
- **Question type**: **choice (multiple-choice)**
- **Correct answer (label)**: D  
- **Model answer (label)**: D  
- **Retrieval path**: local

### 2) Retrieved Evidence
- **Doc1**: The direct conversion of carbon dioxide into biofuels by photosynthetic microorganisms is a promising alternative energy solution. In this study, a model cyanobacterium, Synechococcus elongatus PCC 7942, is engineered to produce free fatty acids (FFA), potential biodiesel precursors, via gene knockout of the FFA-recycling acyl-ACP synthetase and expression of a thioesterase for release of the FFA.
- **Doc2**: Specifically, acyl-CoA/acyl-ACP processing enzymes were targeted to the cytoplasm, peroxisome, or endoplasmic reticulum to generate fatty acid ethyl esters and fatty alkanes with tailored chain length. Activation of endogenous free fatty acids and the subsequent reduction of fatty acyl-CoAs enabled the efficient synthesis of fatty alcohols. Engineering a hybrid fatty acid synthase shifted the free fatty acids to a medium chain-length scale.
- **Doc3**: Past efforts to produce terpenes in photosynthetic organisms have gained only limited success. Here we engineered the cyanobacterium Synechococcus elongatus PCC 7942 to efficiently produce limonene through modeling guided study. Computational modeling of limonene flux in response to photosynthetic output has revealed the downstream terpene synthase as a key metabolic flux-controlling node in the MEP (2-C-methyl-d-erythritol 4-phosphate) pathway-derived terpene biosynthesis.

### 3) Question Understanding
- **Intent**: Identify the **specific enzyme activity knocked out** in *S. elongatus* PCC 7942 to prevent **FFA recycling** (thereby enabling FFA accumulation).
- **Key constraint**: The knockout is described as “FFA-recycling” related.
- **Expected output format**: Choose the correct option label (A/B/C/D) and provide a brief justification.

### 4) Prompt Template Selection
- **Prompt Pool (illustrative)**:
  - **T1**: Binary QA 
  - **T2**: Multiple-choice QA 
- **Selected Template**: **T2 — Multiple-choice QA**
- **Rationale**: The question is explicitly multiple-choice and requires mapping evidence to one option.

### 5) Final Prompt 
```text
[System]
You are a domain expert. Answer strictly based on the provided evidence.
For multiple-choice questions:
1) Identify the key entity/term that answers the question from evidence.
2) Select the corresponding option label (A/B/C/D).
3) Provide a short justification (1–3 sentences) citing evidence as (Doc1), (Doc2), (Doc3).
If only one document is directly relevant, prioritize it and treat others as distractors.

[User]
Question:
Which enzyme activity was knocked out in Synechococcus elongatus PCC 7942 to prevent free fatty acid recycling and enable FFA accumulation?

Options:
(A) ...
(B) ...
(C) ...
(D) ...

Evidence:
(Doc1) "Synechococcus elongatus PCC 7942 ... engineered to produce free fatty acids (FFA) ... via gene knockout of the FFA-recycling acyl-ACP synthetase ..."
(Doc2) "Activation of endogenous free fatty acids ... fatty acyl-CoAs ... (general fatty acid processing)."
(Doc3) "Engineered ... produce limonene ... (terpene pathway)."

Return:
- Option label only on the first line.
- Then a brief evidence-based explanation.

```

### 6) Model Output

D

The evidence states that Synechococcus elongatus PCC 7942 was engineered for free fatty acid production by knocking out the FFA-recalling enzyme acyl-ACP synthetase, described explicitly as “FFA-recycling acyl-ACP synthetase.” Therefore, the correct choice is D. (Doc1)
