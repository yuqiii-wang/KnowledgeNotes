# VirtualBox

Some start guide:

There are two virtualbox windows: host and guest that have diff setting options.

<div style="display: flex; justify-content: center;">
      <img src="imgs/host_vs_guest.png" width="40%" height="40%" alt="host_vs_guest" />
</div>
</br>

To use advanced features such as interactions between host vs guest,
need to install addons both to host and guest.

* To install in host (the extension can be downloaded from Oracle Virtualbox website)

<div style="display: flex; justify-content: center;">
      <img src="imgs/host_vm_add_extension.png" width="30%" height="30%" alt="host_vm_add_extension" />
</div>
</br>

* To install in guest, from "Device" load a CD then double clicked to launch the guest addon CD installation

<div style="display: flex; justify-content: center;">
      <img src="imgs/guest_vm_from_device_insert_cd.png" width="40%" height="40%" alt="guest_vm_from_device_insert_cd" />
</div>
</br>

<div style="display: flex; justify-content: center;">
      <img src="imgs/guest_vm_install_guest_adds.png" width="40%" height="40%" alt="guest_vm_install_guest_adds" />
</div>
</br>

## Some Useful Feature Setup

### Host-Guest Shared Folder Setup

On host config the shared folder:

<div style="display: flex; justify-content: center;">
      <img src="imgs/host_vm_add_shared_folder.png" width="40%" height="40%" alt="host_vm_add_shared_folder" />
</div>
</br>

On guest open the shared folder on "Network"

<div style="display: flex; justify-content: center;">
      <img src="imgs/guest_vm_find_shared_folder_from_network.png" width="30%" height="40%" alt="guest_vm_find_shared_folder_from_network" />
</div>
</br>

### Add USB be Detectable (Also work for adding virtual device)

On host enable USB

<div style="display: flex; justify-content: center;">
      <img src="imgs/host_vm_enable_usb.png" width="43%" height="20%" alt="host_vm_enable_usb" />
</div>
</br>

On host add USB

<div style="display: flex; justify-content: center;">
      <img src="imgs/host_vm_add_usb.png" width="30%" height="20%" alt="host_vm_add_usb" />
</div>
</br>