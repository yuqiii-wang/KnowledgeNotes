# ForgeRock

## OpenAM and OpenDJ Arch

Best architecture practice:
![fr-multi-datastores](imgs/fr-multi-datastores.png "fr-multi-datastores")

### ssoadm

ssoadm is admin console tool for various configuration.

```bash
./ssoadm update-agent -e [realmname] -b [agentname] -u [adminID] -f [passwordfile] -a com.sun.identity.agents.config.agent.protocol=[protocol]
```
