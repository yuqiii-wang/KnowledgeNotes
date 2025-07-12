# GPU Hardware

* TDP: Thermal Design Power (or Thermal Design Point or Thermal Design Parameter), GPU power only 

* TGP: Total Graphics Power, in addition to TDP, include power consumption of graphics memory sub-system + the power consumption of the power circuitry

## Dual GPU Setup and Data Traffic via PCIE

The complete data path:

```txt
GPU1 VRAM -> PCIe 5.0 Bus -> CPU Root Complex -> PCIe 5.0 Bus -> GPU2 VRAM
```

1. The Command (CPU's Job): The application (e.g., PyTorch or TensorFlow) running on the CPU determines the need for the transfer. The CPU issues a command to the driver: "Initiate a DMA transfer of 10GB from GPU1's VRAM at address 0x1111 to GPU2's VRAM at address 0x2222."
2. The Journey Begins (Out of GPU1): The DMA engine on GPU1 reads the 10GB of data from its own ultra-fast VRAM and places it onto the PCIe 5.0 bus. The data begins its journey up the PCIe lanes from the GPU1 slot towards the CPU.
3. The Crossroads (CPU's PCIe Root Complex): The data does not go into the CPU's processing cores or L-caches. Instead, it arrives at a special integrated part of the CPU die called the PCIe Root Complex. This acts as a sophisticated, high-speed traffic switch. The root complex sees the data's destination address (which points to GPU2) and immediately routes the traffic down the PCIe lanes connected to the second GPU's slot.
4. The Final Leg (Into GPU2): The data travels down the second set of PCIe 5.0 lanes and arrives at GPU2. The DMA engine on GPU2 then takes the data off the bus and writes it into the specified address in its own VRAM.
5. Confirmation: Once the transfer is complete, an interrupt is sent to the CPU, letting it know the task is done and it can proceed with the next step of the computation.
