# SOP-07 Inverter reset after grid fault

Revision: 3a (2026-01-12, typo fixed)
Owner: Aylin Demir (aylin.demir@helios-grid.example, +49 40 5550 2233)

## Purpose

This standard operating procedure describes how to reset a string inverter of type SI-60 after the grid protection relay has tripped. It applies to the Nordhafen and Ostfeld sites.

## Preconditions

- The grid operator has confirmed that the fault is cleared.
- No alarm is active on the medium-voltage switchgear.
- The technician holds a valid switching authorisation of level B or higher.

## Procedure

1. Open the inverter monitoring page and note the fault code shown in the event list.
2. Acknowledge the fault on the inverter display with the OK key held for three seconds.
3. Wait until the DC voltage is stable above 600 V for at least sixty seconds.
4. Switch the AC breaker Q2 back on and observe the feed-in for five minutes.
5. Record the reset in the shift log with time, inverter number and fault code.

## Fault codes that must not be reset locally

| Code | Meaning | Escalation |
|---|---|---|
| G31 | Insulation fault on DC side | Call the duty engineer |
| G44 | Earth fault current above 300 mA | Call the duty engineer |
| G52 | Firmware integrity check failed | Open a ticket with the vendor |

## Notes

An inverter that trips more than three times in 24 hours must be taken out of service and inspected. Escalate to the duty engineer Marek Nowak on +49 160 5550 4411.
