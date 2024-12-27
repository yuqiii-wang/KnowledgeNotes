# Virtual File System

Linux treats external devices/peripherals (e.g. Disk, Flash, I/O, Display Monitor) as abstract files so that they can be managed by some unified file operations such as `open`、`close`、`read`、`write`.

The virtual file system is some like the Registry in Windows. It manages the OS configs as well.

## Unix Filesystem

A *filesystem* is a hierarchical storage of data adhering to a specific structure. Filesystems contain files, directories, and associated control information.

Unix systems separate the concept of a file from any associated information about it (such info is called file metadata). Metadata is stored in *inode*.

All the info is stored in the *superblock*. The superblock is a data structure containing information
about the filesystem as a whole. Sometimes the collective data is referred to as *filesystem metadata*


### Superblock

The *superblock* object, which represents a specific mounted filesystem.
Some members are defined as below.

```cpp
struct super_block {
    struct list_head s_list;    /* list of all superblocks */
    dev_t   s_dev;              /* identifier */
    unsigned long   s_blocksize;/* block size in bytes */
    unsigned char s_blocksize_bits; /* block size in bits */
    struct super_operations s_op;   /* superblock methods */
    struct dentry   *s_root;    /* directory mount point */
    struct list_head    s_inodes;   /* list of inodes */
    struct hlist_head   s_anon; /* anonymous dentries */
    struct list_head    s_files;    /* list of assigned files */
};
```

The *super_operations* object, which contains the methods that the kernel can invoke on a specific filesystem.
Some operations are listed below.
```cpp
struct super_operations {
  struct inode *(*alloc_inode)(struct super_block *sb);
  void (*destroy_inode)(struct inode *);
  void (*dirty_inode) (struct inode *);
  int (*write_inode) (struct inode *, int);
  void (*drop_inode) (struct inode *);
  void (*delete_inode) (struct inode *);
  void (*put_super) (struct super_block *);
  void (*write_super) (struct super_block *);
};
```

### Inode

The *inode* object represents all the information needed by the kernel to manipulate a file or directory. The inode object is constructed in memory only as files are accessed or directory are visited.
```cpp
struct inode {
    struct hlist_node   i_hash; /* hash list */
    struct list_head    i_list; /* list of inodes */
    struct list_head    i_sb_list;  /* list of superblocks */
    struct list_head    i_dentry;   /* list of dentries */
    unsigned long       i_ino;      /* inode number */
    struct timespec     i_atime;    /* last access time */
    struct timespec     i_mtime;    /* last modified time */
    struct super_block  *i_sb;      /* associated superblock */
};
```

The inode_operations object, which contains the methods that the kernel can invoke on a specific file.
```cpp
struct inode_operations {
  int (*create) (struct inode *,struct dentry *,int, struct nameidata *);
  struct dentry * (*lookup) (struct inode *,struct dentry *, struct nameidata *);
  int (*link) (struct dentry *,struct inode *,struct dentry *);
  int (*unlink) (struct inode *,struct dentry *);
  int (*symlink) (struct inode *,struct dentry *,const char *);
};
```

### Dentry

The *dentry* object, which represents a directory entry, which is a single component of a path, performing lookup that involves translating each component of a path, ensuring it is valid, and following it to the next component.

Dentry maintains cache for recently visited paths, since often user might re-visit his recently reached paths such as by `cd ..` 

A hash table and hashing function used to quickly resolve a given path into the associated dentry object.
```cpp
struct dentry {
    atomic_t        d_count;        /* usage count */
    struct inode    *d_inode;       /* associated inode */
    struct hlist_node   d_hash;     /* list of hash table entries */
    struct dentry   *d_parent;      /* dentry object of parent */
    struct qstr     d_name;         /* dentry name */
    struct list_head    d_subdirs;  /* subdirectories */
    struct list_head    d_alias;    /* list of alias inodes */
    struct dentry_operations    *d_op;  /* dentry operations table */
    struct super_block  *d_sb;      /* superblock of file */
};
```

The dentry_operations object, which contains the methods that the kernel can invoke on a specific directory entry.
```cpp
struct dentry_operations {
  int (*d_revalidate) (struct dentry *, struct nameidata *);
  int (*d_hash) (struct dentry *, struct qstr *);
  int (*d_compare) (struct dentry *, struct qstr *, struct qstr *);
  int (*d_delete) (struct dentry *);
  void (*d_release) (struct dentry *);
  void (*d_iput) (struct dentry *, struct inode *);
  char *(*d_dname) (struct dentry *, char *, int);
};
```

### File 

The *file* object, which represents an open file as associated with a process. Processes deal directly with files, not superblocks, inodes, or dentries via operations such as `read()` and `write()`.

The file object is the in-memory representation of an open file.The object (but not the physical file) is created in response to the `open()` system call and destroyed in
response to the `close()` system call.

The file object merely represents a process’s view of an open file.The object points back to the dentry (which in turn points back to the inode) that actually represents the open file.The inode and dentry objects, of course, are unique.

```cpp
struct file {
    union {
        struct list_head        fu_list;    /* list of file objects */
        struct rcu_head         fu_rcuhead; /* RCU list after freeing */
    } f_u;
    struct path             f_path;     /* contains the dentry */
    struct file_operations  *f_op;      /* *f_op; */
    mode_t                  f_mode;     /* file access mode */
    struct fown_struct      f_owner;    /* f_owner; */
    struct list_head        f_ep_links; /* list of epoll links */
    spinlock_t              f_ep_lock;  /* epoll lock */
    struct address_space    *f_mapping; /* page cache mapping */
};
```

The file_operations object, which contains the methods that a process can invoke on an open file.
```cpp
struct file_operations {
    struct module *owner;
    ssize_t (*read) (struct file *, char __user *, size_t, loff_t *);
    ssize_t (*write) (struct file *, const char __user *, size_t, loff_t *);
    int (*readdir) (struct file *, void *, filldir_t);
    unsigned int (*poll) (struct file *, struct poll_table_struct *);
    int (*ioctl) (struct inode *, struct file *, unsigned int, unsigned long);
};
```

## `/proc` and `/sys`

There is no real file system exists on `/proc` or `/sys`, but virtual files residing in RAM that helps manage OS config. 

### `/proc`

|File name|Description|
|-|-|
|`/proc/cpuinfo`|Information about CPUs in the system.|
|`/proc/meminfo`|information about memory usage.|
|`/proc/ioports`|list of port regions used for I/O communication with devices.|
|`/proc/mdstat`|display the status of RAID disks configuration.|
|`/proc/kcore`|displays the actual system memory.|
|`/proc/modules`|displays a list of kernel loaded modules.|
|`/proc/cmdline`|displays the passed boot parameters.|
|`/proc/swaps`|displays the status of swap partitions.|
|`/proc/iomem`|the current map of the system memory for each physical device.|
|`/proc/version`|displays the kernel version and time of compilation.|

### `/sys`

`/sys` can be used to get information about your system hardware.

|File/Directory name|Description|
|-|-|
|Block|list of block devices detected on the system like sda.|
|Bus|contains subdirectories for physical buses detected in the kernel.|
|Class|describes the class of device like audio, network, or printer.|
|Devices|list all detected devices by the physical bus registered with the kernel.|
|Module|lists all loaded modules.|
|Power|the power state of your devices.|

### Use example

* `lsof` (list open files) shows Linux running processes by query files

`lsof -i -P` can show current running processes and ports

`lsof | grep 'deleted'` to list deleted files and check what live processes own the file.

* Check network card

If you have multiple network cards, you can run below to config computer to use a particular card

```bash
echo "1" > /proc/sys/net/ipv4/ip_forward
```

To persist the change, you can either

```bash
sysctl -w net.ipv4.ip_forward=1
```

or

```bash
echo "net.ipv4.ip_forward = 1" >> /etc/sysctl.conf
```
