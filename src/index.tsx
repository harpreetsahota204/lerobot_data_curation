import { PluginComponentType, registerComponent } from "@fiftyone/plugins";
import CurationPanel from "./CurationPanel";

registerComponent({
  name: "lerobot_curation_panel", // must match fiftyone.yml's panels entry
  label: "LeRobot Curation",
  component: CurationPanel,
  type: PluginComponentType.Panel,
  activator: ({ dataset }: { dataset: unknown }) => dataset !== null,
  surfaces: "grid",
});
