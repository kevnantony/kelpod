# Kelpod pivot: speaker notes

Generated from `main.tex` (one section per slide, in order). Edit the `\note{}` blocks in `main.tex` and regenerate, rather than editing this file by hand.

## 1. Title slide

Hi everyone. We are Arsha and Kevin, and our project is Kelpod. Kelpod turns old building plans into 3D building models, so energy experts can plan renovations faster. Here is the problem, our first product, and where we are.

## 2. Europe has to renovate its buildings

German and EU rules mean millions of existing buildings need an energy check and a renovation plan. But most old buildings only have paper plans or poor scans. Before an energy expert can do anything, they redraw each plan by hand in their software, which takes three to six hours. The photo is a real old apartment building in Saxony, the kind of building this is about.

## 3. Why today's AI doesn't solve it

There are AI models that read floor plans, but they were trained on American-style plans. German plans use their own symbols for walls, doors and measurements, and the existing models misread them. And they only produce a coloured picture of the plan, which energy software can't use. It needs a proper model with closed walls and rooms.

## 4. First product: old plan in, 3D model out

Our first product takes an old plan and turns it into a 3D model of the building that energy software can open directly. The goal is a few minutes per plan instead of hours, so the expert only checks and corrects the result.

## 5. How it works

Six steps, each handled by its own AI agent. We clean up the scan, find the walls and doors, and read the measurements written on the plan. Then we move the walls so they match those measurements, because the drawing itself is often slightly off. Next we check that everything connects: no gaps in walls, every room closed. Finally we build the 3D model. Where it helps, we also compare against other drawings of the same building. The calculations are done by proven engineering tools; the AI reads and explains.

## 6. Wrong order: heating first

Here is why the plan matters. If an owner buys a bigger heating system first and insulates later, the heating system ends up too big, so they paid twice. With a good building model, Kelpod can work out the right order of measures.

## 7. Where it leads

The 3D model is step one. On top of it we can show where a building loses energy, suggest renovation options with costs, and find the best order within the owner's budget. The picture is a thermal image from NASA: the colours show which homes lose the most heat.

## 8. Our difference

A perfect plan that the building can't take, or that the owner can't pay for, is worth nothing. We show the trade-offs, we are honest about what we don't know, and every number can be traced back to its source.

## 9. Who we serve

We start in Germany, Austria and Switzerland. Our customers are energy consultants, surveyors, architects and housing companies. Two typical users: an independent consultant who wants accurate results without hours of drawing, and a housing company with thousands of flats that needs to process its whole archive at once.

## 10. How we make money

We charge 30 to 80 euros per plan, or a monthly subscription for consulting firms. Larger organisations can license Kelpod and build it into their own software. Customers choose whether it runs in our secure cloud in Germany or on their own computers.

## 11. We work with existing tools

Energy experts already use certified software for the official calculations, and one vendor dominates that market. We don't compete with it. We feed it: Kelpod saves the hours of manual drawing, and the official calculation still happens in the software experts already trust.

## 12. Traction

Our pilot customer is an energy expert. They get an early version for free, running on their own computers. In return we get real plans to train on and their feedback on what the energy software needs. The other numbers are targets for this semester, not results yet.

## 13. Safety first

The software can get a wall wrong. So it always shows what it assumed, and a qualified person makes the final call. Uploaded plans are deleted after processing unless the customer agrees to let us keep them, which keeps us in line with German data protection law.

## 14. Agents as employees

We run the company with AI agents as employees, as the course asks. The founders stay responsible for the decisions and for every claim we make to a customer.

## 15. Semester targets

These are our targets for this semester, not results yet. We want to process ten real buildings, keep a source for every number, get energy estimates within 20 percent of reference data, make sure every suggestion is actually feasible, offer at least three options per building, and cut the preparation time in half.

## 16. Roadmap

Three milestones. First, old plan to 3D model, working on three sample plans. Second, energy estimates and three renovation options per building. By the end of the semester, a demo tested on ten buildings with our pilot partners.

## 17. Next steps

First, we sit down with our pilot expert and learn exactly what their software needs from our models. In parallel we interview consultants and housing companies. And we only train on plans we have permission to use. Thank you, and we are happy to take questions.

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
