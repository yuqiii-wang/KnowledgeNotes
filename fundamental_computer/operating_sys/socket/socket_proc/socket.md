# Socket

Socket in Linux uses inode as implemented in VFS (Virtual File System), and this inode is named *Sockfs*, which is a unique identifier of communication to the launched socket.

When a user launches a socket, socket returns an fd (file descriptor) that associates with an inode.

This inode abstracts lower hardware implementations such as by `ext3` or other device formats.

Declaration and usage
```cpp
#include <sys/socket.h>

// Declare
int socket(int domain, int type, int protocol);

// TCP use example 
int tcp_socket = socket(AF_INET, SOCK_STREAM, 0);
// UDP use example
int udp_socket = socket(AF_INET, SOCK_DGRAM, 0);
```

reference:
http://chenshuo.com/notes/kernel/data-structures/


## Socket structures

```cpp
struct socket {
  socket_state state;
  short type;
  unsigned long flags;
  struct socket_wq __rcu * wq;
  struct file * file;
  struct sock * sk;
  const struct proto_ops * ops;
};  
```
in which,

* `struct socket_wq __rcu * wq`

Wait queue for several uses (rcu: read-copy-update operations)

```cpp
struct socket_wq {
	/* Note: wait MUST be first field of socket_wq */
	wait_queue_head_t	wait;
	struct fasync_struct	*fasync_list;
	unsigned long		flags; /* %SOCKWQ_ASYNC_NOSPACE, etc */
	struct rcu_head		rcu;
} ____cacheline_aligned_in_smp;
```

* `struct sock * sk`

Sock is the lower level implementation of socket that handles individual protocol traffic data parsing.

* `struct file * file`

a file pointer associated with socket, inside it defines file path, operations and inode mapping, etc. 

```cpp
struct file *alloc_file(const struct path *path, fmode_t mode, const struct file_operations *fop)
{
  struct file *file;

  // Many file inits
  file->f_path = *path;
  file->f_inode = path->dentry->d_inode;
  file->f_mapping = path->dentry->d_inode->i_mapping;
  file->f_wb_err = filemap_sample_wb_err(file->f_mapping);
  // ...

  return file;
}
```

## Sock creation

`socket()` implementation is in `net/socket.c` in the linux kernel sources, where all code starts.
```cpp
SYSCALL_DEFINE3(socket, int, family, int, type, int, protocol);
```

Socket creation first launches an inode and inits a sock with a specified protocol.
```cpp
int __sock_create(struct net *net, int family, int type, int protocol, struct socket **res, int kern)
{
  struct socket *sock;
  sock = sock_alloc(); // allocate a sock
  // ...

  int err = pf->create(net, sock, protocol, kern); // create a sock with the specified protocol
  // ...
}
```

Inside a socket defines an inode and a sock. 
```cpp
struct socket *sock_alloc(void)
{
  struct inode *inode;
  struct socket *sock;

  // Many inits for inode and sock
  // ...
}
```

`pf` is an obj of `struct net_proto_family` that creates a sock for a designated protocol.
```cpp
struct net_proto_family {
  int  family;
  int  (*create)(struct net *net, struct socket *sock,
      int protocol, int kern);
  struct module *owner;
};
```

## `inode`

This example uses `ext4_alloc_inode`. Inside, `vfs_inode` is a vfs inode that abstracts lower level `ext4` implementation.
```cpp
static struct inode *ext4_alloc_inode(struct super_block *sb){
  struct ext4_inode_info *ei;

  ei = kmem_cache_alloc(ext4_inode_cachep, GFP_NOFS);
  if (!ei)
    return NULL;

  // Many ei configs
  // ...

  return &ei->vfs_inode;
}
``` 

The materialization on disk for ext4.
```cpp
struct ext4_inode {
 __le16 i_mode;  /* File mode */
 __le16 i_uid;  /* Low 16 bits of Owner Uid */
 __le32 i_size_lo; /* Size in bytes */
 __le32 i_atime; /* Access time */
 __le32 i_ctime; /* Inode Change time */
 __le32 i_mtime; /* Modification time */
 __le32 i_dtime; /* Deletion Time */
 __le16 i_gid;  /* Low 16 bits of Group Id */
 __le16 i_links_count; /* Links count */
 __le32 i_blocks_lo; /* Blocks count */
 __le32 i_flags; /* File flags */
 // ......
}
```

How OS understands the ext4 disk format by reading this `ext4_inode_info` config.
```cpp
struct ext4_inode_info {
 __le32 i_data[15]; /* unconverted */
 __u32 i_dtime;
 ext4_fsblk_t i_file_acl;
 ......
};
```

`inode` is the abstraction APIs for many lower level implementation. Each lower level implementation can register its unique `file_operations` to `struct inode`. 
```cpp
struct inode {
  umode_t                 i_mode;
  unsigned short          i_opflags;
  kuid_t                  i_uid;
  kgid_t                  i_gid;
  unsigned int            i_flags;

  const struct inode_operations   *i_op;
  struct super_block      *i_sb;
  struct address_space    *i_mapping;

  const struct file_operations    *i_fop; /* former ->i_op->default_file_ops */

  // ...
}
```