import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";

const LOADER_NODE = "LLMModelLoader";

app.registerExtension({
  name: "CausalTile.LLMModelLoader",

  async nodeCreated(node) {
    if (node.comfyClass !== LOADER_NODE) return;

    const modelWidget = node.widgets?.find((w) => w.name === "model");
    if (!modelWidget) return;

    node.addWidget("button", "⟳ 刷新模型列表", null, async () => {
      const read = (name) => node.widgets.find((w) => w.name === name)?.value ?? "";
      try {
        const res = await api.fetchApi("/causaltile/llm/models", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            base_url: read("base_url"),
            api_key: read("api_key"),
            headers: read("headers"),
          }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || res.statusText);

        const models = data.models || [];
        if (!models.length) throw new Error("接口没有返回任何模型");

        modelWidget.options.values = models;
        if (!models.includes(modelWidget.value)) modelWidget.value = models[0];
        node.setDirtyCanvas(true, true);
        console.log(`[CausalTile] 已获取 ${models.length} 个模型`);
      } catch (err) {
        console.error("[CausalTile] 刷新模型列表失败:", err);
        alert("刷新模型列表失败：" + err.message);
      }
    });
  },
});
