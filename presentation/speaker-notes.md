# Kelpod pivot: speaker notes

Generated from `main.tex` (one section per slide, in order). Edit the `\note{}` blocks in `main.tex` and regenerate, rather than editing this file by hand.

## 1. Title

Hi everyone. We are Arsha and Kevin, and our project is Kelpod. In one sentence: Kelpod turns incomplete building information into a renovation roadmap an owner can actually afford and implement. Here is the problem, what we built, and where we are.

## 2. Europe has to renovate its buildings

Start with the building, not the software. Many owners decide on the heating system first, and then find the building still loses most of its heat. The order of work matters as much as the measures themselves. This photo is a real multi-family house in Saxony that is a listed monument, so every renovation has to work around its protection status.

## 3. Wrong order: heating first

If you size a new heating system first, you pay for a system that is too big once the house is insulated. The right order is the measures that reduce demand first. Kelpod works out that order.

## 4. Kelpod: uploads in, roadmap out

Users give us what they already have: plans, photos, bills and records. Kelpod turns this into a building model and a roadmap. Every number on the roadmap can be traced back to where it came from.

## 5. What Kelpod does

Seven steps from upload to a plan a professional can review. Steps 1 to 3 build the building record. Steps 4 to 6 compare options. Step 7 is the output, and a qualified person still signs it off. The picture is a NASA thermograph from 1977: the colours show which homes lose the most heat. This is the question Kelpod answers for a whole estate, not one house at a time.

## 6. Our difference

This is the claim that sets us apart. A theoretically perfect measure that the building cannot take, or that the owner cannot pay for, is worth nothing. We show the trade-offs, and we show what we do not know.

## 7. Who we serve

Our first market is German multi-family housing. The buyers are the people who manage these buildings and decide on the money. The impact is measurable: energy, CO\textsubscript{2

## 8. How we make money

Three revenue streams: one-off assessments for a single building, a subscription for portfolios, and licences for organisations that want to build Kelpod into their own tools.

## 9. We complement, we don't replace

We are not competing with certified calculation tools, and we say so. Our value is the planning layer on top: which measure first, what it costs, and what is still uncertain. The calculation tools in the market are Hottgenroth / ETU and ENVISYS EVEBI.

## 10. Traction

Be precise about this. One customer is locked in for the pilot. The other numbers are targets for this semester, not results yet. The pilot partners should give feedback from at least five end users.

## 11. Six agents, one workflow

Six agents pass structured data along one workflow. The important design choice: the calculations come from deterministic engineering tools, and the language and vision models only extract information and explain it. Planned stack: Python with FastAPI, PostgreSQL with PostGIS, OCR and vision-language models, EnergyPlus with OpenStudio, and pymoo for optimisation.

## 12. Safety first

This is the principle we will not compromise on. The software can be wrong about a wall. So it must always show what it assumed, and a professional makes the final call.

## 13. Agents as employees

We run the company with agents as employees, the way the course asks. The founders stay responsible for the decisions and for every claim we make to a customer.

## 14. Semester targets

These are our targets for the semester. They are development targets, not validated product claims. Completeness: at least 10 residential buildings with at least 85 percent of required attributes extracted or flagged. Speed: at least 50 percent less first-pass preparation time than the manual baseline. We will report against each of them at the end.

## 15. Roadmap

The semester splits into three milestones. The first gives us an upload pipeline that works on three sample buildings. The second gives us the energy engine and scenario comparison. The end goal is a demo that we test on ten buildings and review with two pilot partners.

## 16. Next steps

Our next step is listening, not building. We interview the people who manage these buildings before we fix the workflow. BIM reconstruction, automated procurement and pan-European regulation come later and are out of scope for this semester.
