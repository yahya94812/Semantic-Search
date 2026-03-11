# Meeting Notes — Berlin Startup Partnership

**Date:** March 2020
**Attendees:** Sarah Koch (CEO, NeuroGrid GmbH), Tom Bauer (CTO), and our team

## Overview

We met with NeuroGrid, a Berlin-based AI startup working on neural interface hardware.
They are looking for a software partner to build the data pipeline layer for their device.

## Discussion Points

- NeuroGrid has raised €4M seed funding from a German VC firm (HV Capital).
- Their hardware prototype captures 64-channel EEG signals at 2kHz sampling rate.
- They need a real-time streaming pipeline capable of processing ~500MB/s of raw sensor data.
- Latency requirements are strict: end-to-end under 20ms for closed-loop feedback scenarios.

## Technical Requirements Discussed

1. Kafka-based ingestion layer for raw EEG streams.
2. Python signal processing microservices using MNE and SciPy.
3. WebSocket API for real-time dashboard updates.
4. Cloud storage on AWS S3 with Parquet format for long-term archival.

## Partnership Model

- **Revenue share:** 12% of enterprise license fees.
- We retain IP rights to the pipeline software.
- **Exclusivity window:** 18 months in the neurotechnology vertical.

## Action Items

- [ ] Send technical proposal by 15 March 2020.
- [ ] Schedule follow-up call with their lead engineer, Klaus Müller.
- [ ] Draft NDA and forward to legal team.

**Next Meeting:** 22 March 2020 via Zoom.
