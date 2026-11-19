# Kelpod pivot: speaker notes (5-minute version)

Generated from `main.tex` (one section per slide, in order). **[click]** means: advance one step in the PDF. Edit the `\note{}` blocks in `main.tex` and regenerate, rather than editing this file by hand.

## 1. Title slide

Hi everyone. We are Arsha and Kevin, and this is Kelpod. We turn old building plans into 3D models that energy software can use.

## 2. The problem

Germany has to renovate a huge number of old buildings, like this one in Saxony.

**[click]** The building energy law means millions of them need an energy audit.

**[click]** But most old buildings only have old PDF scans of their plans.

**[click]** So an energy expert redraws every plan by hand in HottCAD, the standard energy software. That takes three to six hours, before the real work even starts.

## 3. Today's AI can't do it

Why not just use AI?

**[click]** The existing floor-plan models were trained on American plans.

**[click]** German plans follow their own drawing standard, and those models misread it.

**[click]** And they only give you a coloured picture. Energy software needs a real model with closed walls and rooms.

## 4. Kelpod: old plan in, 3D model out

That's what Kelpod does. You give it the old scan.

**[click]** Kelpod reads it.

**[click]** And produces a 3D model in IFC, the standard building-model format, which HottCAD opens directly.

**[click]** Our goal is under three minutes per plan instead of three to six hours. The expert only checks the result.

## 5. How it works

Six steps. We clean the scan.

**[click]** Find the walls, doors and windows.

**[click]** Read the measurements written on the plan.

**[click]** Move the walls so they match those measurements, because the drawing itself is often slightly off.

**[click]** Check that everything connects: no gaps, every room closed.

**[click]** And export the 3D model.

**[click]** Each step is its own AI agent.

## 6. Integrate first, then replace

Our strategy has two phases. Today the expert starts from an old plan.

**[click]** Kelpod builds the model, and it goes into HottCAD for the official calculation. That saves hours and earns us trust and real data.

**[click]** Later, Kelpod does the official calculation itself, and the expert no longer needs HottCAD. That needs a certified calculation engine, so we start with the part we can win now.

## 7. Customers and revenue

Who pays? Energy-efficiency experts, surveyors, architects, and housing companies with thousands of flats.

**[click]** We charge 30 to 80 euros per plan, or a monthly licence, and larger clients can build Kelpod into their own software.

## 8. Traction and roadmap

One pilot customer is locked in: an energy expert. They get Kelpod for free, and we get real plans and feedback. We build the company with AI agents too: coding, research and sales agents, with the founders making the decisions.

**[click]** By week 6: plan to IFC model on three sample plans.

**[click]** By week 10: energy estimates and three renovation options per building.

**[click]** By the end of the semester: a demo on ten buildings. Thank you, we are happy to take questions.

## If someone asks: the technical terms

Not on the slides on purpose. Use these only if a question goes deeper.

- **"Experts' software"** is HottCAD by Hottgenroth / ETU, the dominant tool for the official energy calculations in Germany. ENVISYS EVEBI is another one.
- **"New laws"** are the German Building Energy Act (GEG) and EU rules; the renovation plan is called an iSFP (individual renovation roadmap).
- **"German drawing symbols"** are defined in the standard DIN 1356-1: wall hatching, door swings, window marks.
- **"Measurements"** are dimension chains (*Maßketten*), sill heights (e.g. BRH 87.5) and room labels (e.g. Wohnen, 14.5 m²).
- **"3D model"** is an IFC file (versions 2x3 and IFC4), a standard building-model format. We also export DWG/DXF and gbXML.
- **"Today's AI"** means floor-plan models like DeepFloorPlan and RoomFormer. We fine-tune them on German plans.
- **"Fit walls to them"**: the measurements become constraints in a geometric optimisation. **"Check it all connects"**: walls, doors and rooms form one spatial graph, so gaps and open rooms show up.
- **"Energy estimates within 20%"** is a CV(RMSE) of at most 20% against reference buildings.
- **"Our own computers"** means an on-premise Docker container on the customer's NVIDIA GPUs; the cloud runs in Frankfurt under GDPR.
- **Tech stack:** Next.js, FastAPI, fine-tuned DeepFloorPlan / RoomFormer, OCR (Tesseract or Donut / TrOCR), Shapely for geometry, IfcOpenShell for the model file.
- **"Replace HottCAD later"** means building our own engine for the official GEG energy calculation. That calculation must be certified, so phase 2 is a large, regulated project; phase 1 (feeding HottCAD) is what earns the trust, data and revenue to get there.
