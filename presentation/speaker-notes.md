# Kelpod pivot: speaker notes

Generated from `main.tex` (one section per slide, in order). **[click]** means: advance one step in the PDF. Edit the `\note{}` blocks in `main.tex` and regenerate, rather than editing this file by hand.

## 1. Title slide

Hi everyone. We are Arsha and Kevin, and our project is Kelpod. Kelpod turns old building plans into 3D building models, so energy experts can plan renovations faster. Here is the problem, our first product, and where we are.

## 2. Europe has to renovate its buildings

Europe has to renovate a huge number of old buildings.

**[click]** German and EU rules mean millions of them now need an energy check and a renovation plan.

**[click]** But most old buildings only have paper plans or poor scans.

**[click]** So before an energy expert can do anything, they redraw each plan by hand. That takes three to six hours per plan.

**[click]** That is the bottleneck: hours of drawing before the real work even starts. The photo is a real old apartment building in Saxony, the kind of building this is about.

## 3. Why today's AI doesn't solve it

You might think AI already solves this. It doesn't, for three reasons.

**[click]** The AI models that read floor plans were trained on plans from other countries, mostly the US.

**[click]** German plans use their own symbols for walls, doors and measurements, and those models misread them.

**[click]** And they only produce a coloured picture of the plan. Energy software can't use a picture; it needs a proper model with closed walls and rooms.

## 4. First product: old plan in, 3D model out

So here is our first product. It starts with an old plan, on paper or scanned.

**[click]** Kelpod reads the plan and rebuilds the building.

**[click]** Out comes a 3D model that energy software can open directly.

**[click]** The goal: a few minutes instead of hours. The expert only checks and corrects, instead of redrawing.

## 5. How it works

How does it work? Six steps, each handled by its own AI agent. First, we clean up the scan.

**[click]** Then we spot the walls and doors.

**[click]** We read the measurements written on the plan.

**[click]** We move the walls so they match those measurements, because the drawing itself is often slightly off.

**[click]** We check that everything connects: no gaps in walls, every room closed.

**[click]** And we build the 3D model.

**[click]** Each step is an AI agent, but the actual calculations are done by proven engineering tools. The AI reads and explains; it doesn't invent numbers.

## 6. Wrong order: heating first

Why does the building model matter so much? Here is a common mistake. An owner buys a bigger heating system first.

**[click]** And insulates later.

**[click]** Now the heating system is too big for the insulated house. They paid twice.

**[click]** With a good building model, Kelpod can put the measures in the right order.

## 7. Where it leads

That's where this leads.

**[click]** The 3D model is our first product.

**[click]** On top of it, we can show where a building loses energy.

**[click]** Suggest renovation options with their costs.

**[click]** And find the best order within the owner's budget.

**[click]** The end goal: our own certified energy calculation, so experts need only one tool. The picture is a NASA thermal image: the colours show which homes lose the most heat.

## 8. Our difference

What makes us different: we recommend what can actually be built, not a perfect plan on paper.

**[click]** If we are unsure about something, we show it instead of hiding it.

**[click]** We tell the user when an expert needs to look.

**[click]** And every number shows where it came from.

## 9. Who we serve

Who do we serve?

**[click]** Energy consultants and building surveyors.

**[click]** Architects and housing companies. We start in Germany, Austria and Switzerland.

**[click]** Two typical users. An independent consultant who wants accurate results without hours of drawing.

**[click]** And a housing company with thousands of flats that needs to process its whole archive at once.

**[click]** The impact: faster energy checks, so more buildings get renovated, and less CO2.

## 10. How we make money

How do we make money? First, per plan: 30 to 80 euros each.

**[click]** Second, a monthly subscription for consulting firms.

**[click]** Third, larger clients can license Kelpod and build it into their own software.

**[click]** And customers choose whether it runs in our secure cloud in Germany or on their own computers.

## 11. Integrate first, then replace

Our strategy has two phases. Today an energy expert starts from an old plan.

**[click]** Kelpod turns it into a model.

**[click]** And that model goes into HottCAD, the software experts already use for the official calculation. This saves them hours, and it earns us their trust and real data.

**[click]** Later, Kelpod does the official calculation itself, so the expert no longer needs a separate tool. We go from feeding HottCAD to replacing it.

**[click]** That second phase is the big one: the official calculation has to be certified. So we start with the part we can win now, and build towards the rest.

## 12. Traction

Where are we? One customer is locked in for our pilot: an energy expert.

**[click]** They get an early version for free, running on their own computers.

**[click]** In return we get real plans to learn from and their feedback on what their software needs.

**[click]** Our targets for this semester are 15 interviews, 2 pilot partners and 3 conversations about pricing. These are targets, not results yet.

## 13. Safety first

Safety comes first. The AI suggests, experts decide. Our results are not official measurements.

**[click]** If information is missing, we flag it instead of guessing.

**[click]** A person approves every result.

**[click]** And uploaded plans are deleted after use, unless the customer agrees to let us keep them. That keeps us in line with German data protection law.

## 14. Agents as employees

We also run the company with AI agents as employees, as the course asks.

**[click]** Coding agents build the product and its tests.

**[click]** Research agents prepare interviews and find data.

**[click]** Sales agents keep track of our pilot customers.

**[click]** But the founders decide: technical choices, customers, and every claim we make.

## 15. Semester targets

Our targets for this semester.

**[click]** Process ten real buildings.

**[click]** Every number shows its source.

**[click]** Energy estimates within 20 percent of reference data.

**[click]** Every suggestion passes budget and technical checks.

**[click]** At least three options per building.

**[click]** And half the preparation time. Again: these are targets, not results.

## 16. Roadmap

Our roadmap has three milestones.

**[click]** By week 6: old plan to 3D model, working on three sample plans.

**[click]** By week 10: energy estimates and three renovation options per building.

**[click]** By the end of the semester: a demo tested on ten buildings with our pilot partners.

## 17. Next steps

What's next?

**[click]** First, we sit down with our pilot expert and learn exactly what their software needs.

**[click]** We interview consultants and housing companies.

**[click]** We only train on plans we have permission to use.

**[click]** And one honest reminder: all the numbers you saw today are targets, not results yet. Thank you, and we are happy to take questions.

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
