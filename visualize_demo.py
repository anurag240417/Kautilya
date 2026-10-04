import matplotlib.pyplot as plt
import networkx as nx

from backend.domain.graph import AddrTxEdge, TxAddrEdge, TxTxEdge, AddrAddrEdge
from backend.graph.builder import KautilyaGraph

# Create an empty graph
cg = KautilyaGraph()

# Create a small mock scenario:
# Wallet A sends to Tx 1
# Tx 1 sends to Wallet B and Wallet C
# Wallet C sends to Tx 2
# Tx 2 sends to Tx 3 (direct transaction to transaction money flow)
# Wallet A also has a direct address-to-address relationship with Wallet D

# 1. Address -> Tx Edges
cg.add_addr_tx_edges([
    AddrTxEdge(input_address="Wallet A", txid=1, is_synthetic=False),
    AddrTxEdge(input_address="Wallet C", txid=2, is_synthetic=False),
])

# 2. Tx -> Address Edges
cg.add_tx_addr_edges([
    TxAddrEdge(txid=1, output_address="Wallet B", is_synthetic=False),
    TxAddrEdge(txid=1, output_address="Wallet C", is_synthetic=False),
])

# 3. Tx -> Tx Edges
cg.add_tx_tx_edges([
    TxTxEdge(source_txid=2, target_txid=3, is_synthetic=False),
])

# 4. Address -> Address Edges
cg.add_addr_addr_edges([
    AddrAddrEdge(input_address="Wallet A", output_address="Wallet D", is_synthetic=False),
])

G = cg.G

# Define layout
pos = nx.spring_layout(G, seed=42)

# Extract nodes by type
tx_nodes = [node for node, attr in G.nodes(data=True) if attr.get("type") == "transaction"]
wallet_nodes = [node for node, attr in G.nodes(data=True) if attr.get("type") == "wallet"]

# Set up the plot
plt.figure(figsize=(10, 6))

# Draw Transaction nodes (Blue squares)
nx.draw_networkx_nodes(G, pos, nodelist=tx_nodes, node_color='lightblue', node_shape='s', node_size=1500, label="Transactions")

# Draw Wallet nodes (Green circles)
nx.draw_networkx_nodes(G, pos, nodelist=wallet_nodes, node_color='lightgreen', node_shape='o', node_size=1500, label="Wallets")

# Draw edges
nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20, edge_color='gray', connectionstyle="arc3,rad=0.1")

# Draw labels (we prepend 'Tx' to transaction IDs so they look good in the plot)
labels = {}
for node in G.nodes():
    if G.nodes[node]["type"] == "transaction":
        labels[node] = f"Tx {node}"
    else:
        labels[node] = str(node)
nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight="bold")

plt.title("Kautilya Graph Representation Demo")
plt.legend(scatterpoints=1)
plt.axis("off")

# Save directly to the artifacts directory
image_path = "/Users/krish/.gemini/antigravity-ide/brain/e35ef573-e04a-4db7-9bfe-a4f6886ad96e/graph_demo.png"
plt.tight_layout()
plt.savefig(image_path, dpi=150, bbox_inches='tight')
print(f"Saved graph visualization to {image_path}")