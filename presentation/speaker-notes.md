# Kelpod pivot: speaker notes

Generated from `main.tex` (one section per slide, in order). Edit the `\note{}` blocks in `main.tex` and regenerate, rather than editing this file by hand.

## 1. Title slide

Hi everyone. We are Arsha and Kevin, and our project is Kelpod. Kelpod turns old German floor plans into 3D building models that energy software can use, and builds renovation roadmaps on top. Here is the problem, our first product, and where we are.

## 2. Europe has to renovate its buildings

Because of the German Building Energy Act (GEG) and EU rules, millions of existing buildings need digital energy audits, heat-load calculations and renovation roadmaps (iSFP). But most old buildings only have degraded PDF scans or 1970s blueprints. Our project plan puts this at over 80 percent; we still need a source before we state that number on a slide. Highly paid engineers spend 3 to 6 hours redrawing each plan into HottCAD by hand. The photo is a real listed multi-family house in Saxony, the kind of building this applies to.

## 3. Why today's AI fails on German plans

Models like DeepFloorplan and RoomFormer use room-boundary attention and work on US-style plans. German solid construction (Massivbau) plans are different: DIN 1356-1 symbols, multi-layer masonry hatching, and dimension chains that are often rotated. On top of that, these models output pixel masks, and BIM software crashes on them unless they are turned into closed vector geometry.

## 4. First product: scan in, 3D model out

Our first product turns an old floor plan into a 3D building model in IFC, versions 2x3 and IFC4, which HottCAD imports directly. We also export DWG or DXF, and gbXML for other energy simulation tools. The goals are more than 95 percent closed geometry and under three minutes per plan, on a local GPU or in the cloud. The engineer verifies and tweaks instead of redrawing from scratch. This is the first building block of the full renovation platform.

## 5. How it works

Step 1 straightens the scan, removes stains and fold marks, and fixes the scale. Step 2 uses DeepFloorPlan and RoomFormer fine-tuned on DIN 1356-1 plans. DeepFloorPlan already uses attention for room boundaries; we want to adapt it to interrupted wall lines, junctions and unclear boundaries in old German plans. For RoomFormer we want stronger geometric links between neighbouring corners, so the layouts come out consistent. Step 3 is OCR: dimension chains, sill heights like BRH 87.5, and room labels. Step 4 is the key idea: detected walls can still sit in the wrong place, so we attach each dimension annotation to its wall segment and use them as constraints in a geometric optimisation. Step 5 treats walls, openings and rooms as one connected spatial graph instead of separate pieces, which lets us find disconnected walls and invalid room boundaries and compare alternative layouts before we build the model. Step 6 builds the IFC file with IfcOpenShell. On top of that, floor plans, sections and elevations describe the same building, so matching them lets us cross-check dimensions, find missing information, and tell real building changes apart from errors in the drawings. These are ideas we will test with our pilot expert's plans; they are not built yet.

## 6. Wrong order: heating first

If you size a new heating system first, you pay for a system that is too big once the house is insulated. The right order is the measures that reduce demand first. Kelpod works out that order.

## 7. Where it leads: the full platform

The floor-plan model is step one. On top of it, the platform simulates where energy is lost, builds costed renovation packages, finds the best order of work under budget and regulatory constraints, and produces a phased plan for an expert to review. The picture is a NASA thermograph from 1977: the colours show which homes lose the most heat.

## 8. Our difference

This is the claim that sets us apart. A theoretically perfect measure that the building cannot take, or that the owner cannot pay for, is worth nothing. We show the trade-offs, and we show what we do not know.

## 9. Who we serve

Our market is the DACH region: energy-efficiency experts (Energieeffizienz-Experten), building surveyors, architects and housing associations. We design for two users. The independent expert is highly technical, hates manual CAD tracing, cares about data sovereignty and runs local hardware; they need imports that are more than 95 percent accurate. The draftsman at a large housing association manages over 10,000 units and needs to batch-process archive PDFs into IFC files for portfolio-wide ESG reporting.

## 10. How we make money

We charge 30 to 80 euros per scanned plan, or a monthly licence for consulting firms, and API or white-label licences for larger organisations. Customers choose where it runs: our cloud API on GDPR-compliant servers in Frankfurt, or an on-premise Docker container on their own NVIDIA GPUs for privacy-first clients.

## 11. We feed HottCAD, we don't replace it

Hottgenroth / ETU, with HottCAD, holds a near-monopoly on government-certified compliance calculations. ENVISYS EVEBI is another established tool. We do not compete with them. We feed them: Kelpod replaces the hours of manual redrawing, and the certified calculation still happens in their software.

## 12. Traction

Our pilot partner is an energy-efficiency expert. We install an alpha version as a Docker container on their own GPU server, for free, to remove their manual-entry bottleneck. In return we train our OCR and segmentation on their archive of real plans, and use their expertise to map exactly which IFC attributes HottCAD needs. Once imports work without errors, we package the engine as a paid API for other energy consulting firms. The other numbers are targets for this semester, not results.

## 13. Six agents, one workflow

Six agents pass structured data along one workflow. The important design choice: the calculations come from deterministic engineering tools, and the language and vision models only extract information and explain it. Planned stack: Python with FastAPI, PostgreSQL with PostGIS, OCR and vision-language models, EnergyPlus with OpenStudio, and pymoo for optimisation.

## 14. Safety first

The software can be wrong about a wall. So it must always show what it assumed, and a professional makes the final call. On the cloud tier we keep nothing: uploaded plans are deleted after processing unless the customer explicitly opts in to help with training. That is part of being compliant with the German GDPR, the DSGVO.

## 15. Agents as employees

We run the company with agents as employees, the way the course asks. The founders stay responsible for the decisions and for every claim we make to a customer.

## 16. Semester targets

These are our targets for the semester. They are development targets, not validated product claims. Completeness: at least 10 residential buildings with at least 85 percent of required attributes extracted or flagged. Speed: at least 50 percent less first-pass preparation time than the manual baseline. We will report against each of them at the end.

## 17. Roadmap

The semester splits into three milestones. The first gives us the scan-to-3D pipeline working on three sample plans. The second gives us the energy engine and scenario comparison. The end goal is a demo that we test on ten buildings and review with two pilot partners.

## 18. Next steps

First we sit down with our pilot expert and map exactly which IFC attributes HottCAD needs for an error-free import. In parallel we interview experts, surveyors and housing associations. We train only on real plans we have permission to use, and we check data licensing before any commercial use.
