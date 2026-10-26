# adpython

Tools for the ASR 920 to NCS 540 migration.

## Migration topology

`migration-topology` reads a JSON description of a device migration and writes an SVG topology diagram. A null port is labeled **unlabeled**. A null move order is labeled **not recorded**. The renderer does not fill in missing interface names, addresses, or service instances.

### Render the sample

```bash
PYTHONPATH=src python3 -m migration_topology render \
  samples/asr920-to-ncs540.json \
  -o samples/asr920-to-ncs540.svg
```

The sample is a like-for-like move from a Cisco ASR 920 (IOS XE) to a Cisco NCS 540 (IOS XR). Both ethernet uplinks are unlabeled because the port map is not available, and the diagram is marked blocked on the running-config and the port map. Ordered stages are the uplink move, IS-IS, SR-MPLS, then VPN services (L2VPN/L3VPN). Cisco Crosswork Network Controller (CNC) is the controller. Crosswork Workflow Manager (CWM) runs the stages. NSO commits each stage template.

Install the command as `migration-topology` with `pip install -e .` if you want it on `PATH`.

### Description

The JSON object has these fields:

| Field | Meaning |
| --- | --- |
| `source`, `target` | Device `name` and `os`. Optional `vendor`. Set `nso_device_name` to `null` when the NSO device name is unknown. |
| `links` | Each link has `name`, `source_port`, `target_port`, and `move_order`. Use `null` for an unknown port or move order. |
| `network` | Label for the far end of the uplinks. Optional `note` when peers are unknown. |
| `controller` | `name`, plus `workflow_manager` and `nso`. Each component has `name` and `role`. |
| `stages` | At least one stage, with `order`, `name`, and `template_kind`. Drawn in `order`. |
| `blocked_on` | Gaps the diagram should state, such as a missing running-config or port map. |
| `kind`, `status`, `title` | Optional. `status` of `blocked` is shown on the diagram. |

### Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```
