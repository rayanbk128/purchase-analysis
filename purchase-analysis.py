import streamlit as st
import pandas as pd
import math
from math import sqrt
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px
import plotly.graph_objects as go

# Set Streamlit page configuration
st.set_page_config(
    page_title="Mini Project ",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Original product positions
product_positions = {
    'A': (2, 3),
    'B': (5, 1),
    'C': (7, 8),
    'D': (1, 6),
    'E': (4, 4),
}

# Keep a copy of original positions for comparison
original_product_positions = product_positions.copy()


def euclidean(p1, p2):
    return sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


# Function to optimize store layout based on frequent itemsets
def optimize_store_layout(frequent_itemsets, product_positions):
    

    # Copy of original positions
    new_positions = product_positions.copy()

    # Sort itemsets: by length descending, then support descending
    sorted_itemsets = frequent_itemsets.sort_values(['length', 'support'], ascending=[False, False])

    # Track placed products
    placed_products = set()

    # Final result containers
    consolidated_positions = {}
    consolidated_mapping = {}
    combined_products = []

    #  Euclidean distance to exit point (0,0)
    def distance_to_exit(pos):
        return math.sqrt(pos[0] ** 2 + pos[1] ** 2)

    # Process each itemset (length ≥ 2)
    for _, row in sorted_itemsets.iterrows():
        itemset = list(row['itemsets'])

        # Skip itemsets that don’t offer new items
        remaining_items = [item for item in itemset if item not in placed_products]
        if len(remaining_items) < 2:
            continue

        # Use the closest unplaced item to the exit (0,0) as reference position
        closest_item = min(remaining_items, key=lambda item: distance_to_exit(product_positions[item]))
        placement_position = product_positions[closest_item]

        # Generate a combined name (based on all items in itemset, even placed)
        combined_name = ''.join(sorted(remaining_items))

        # Mark unplaced items as used and map them to the new group
        for item in remaining_items:
            consolidated_mapping[item] = combined_name
            placed_products.add(item)

        # Save final group
        consolidated_positions[combined_name] = placement_position
        combined_products.append((combined_name, placement_position, remaining_items))

    # Handle ungrouped frequent items (length == 1)
    for _, row in sorted_itemsets[sorted_itemsets['length'] == 1].iterrows():
        item = list(row['itemsets'])[0]
        if item not in placed_products:
            consolidated_positions[item] = product_positions[item]
            placed_products.add(item)

    # Keep untouched positions for completely unfrequent items
    for item in product_positions:
        if item not in placed_products:
            consolidated_positions[item] = product_positions[item]

    return consolidated_positions, consolidated_mapping, combined_products


# Custom container class
def styled_container():
    # Return a styled container for consistent UI elements
    return st.container()
#for dsiplay and hide the content of the button
if "show_store_layout" not in st.session_state:
    st.session_state.show_store_layout = False

  # Define the toggle function
def toggle_store_layout():
    st.session_state.show_store_layout = not st.session_state.show_store_layout


# App Header
st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1>Mini Project Data Mining</h1>
        <p>Analyze purchasing patterns and optimize store layout</p>
    </div>
    <hr>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h3 style='text-align: center;'>Controls & Settings</h3>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # File upload
    st.markdown("<h4>Data Import</h4>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload transaction data", type=["csv", "xlsx"])

    if uploaded_file:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h4>Analysis Parameters</h4>", unsafe_allow_html=True)
        min_support = st.slider("Minimum Support", 0.01, 0.5, 0.2, 0.01,
                                help="Minimum support threshold for frequent itemsets")

        #min_itemset_size = st.slider("Min Itemset Size for Consolidation", 2, 5, 2,
                                    # help="Minimum number of items to consider for consolidation")

        
        
        max_itemsets = 20
        optimize_layout = True

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center; padding-top: 1rem;">
            <p style="font-size: 0.8rem;"><b>Developed by :</b> Laouici & Boukabous</p>
        </div>
    """, unsafe_allow_html=True)

# Main content
if uploaded_file:
    try:
        # Load data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        with styled_container():
                st.markdown("<h3>Dataset Overview :</h3>", unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True)
                df1 = df.copy()

                cols = st.columns(3)
                cols[0].metric("Total Transactions", f"{len(df)}")

                # Extract product columns
                product_cols = [col for col in df.columns if col != 'transaction_ID' and col != 'Distance']
                cols[1].metric("Products", f"{len(product_cols)}")

                # Calculate purchases per transaction
                if len(product_cols) > 0:
                    avg_items = df[product_cols].sum(axis=1).mean()
                    cols[2].metric("Avg Items per Transaction", f"{avg_items:.2f}")

        #create the diagram
        with styled_container():
             hide_button_text = """
                <style>
                .stButton > button {
                   
                   border-style: solid;
                   border-width: 3px;
                    border-radius: 15px;
                    padding: 10px 15px;
                     transition-duration: 0.3s;
                   
                     }
                     </style>
                        """
             st.markdown(hide_button_text, unsafe_allow_html=True)
             if st.button("📊 Show Store Layout", on_click=toggle_store_layout):
              if st.session_state.show_store_layout:
                # Create store layout visualization
                layout_df = pd.DataFrame({
                    'Product': list(product_positions.keys()),
                    'X': [pos[0] for pos in product_positions.values()],
                    'Y': [pos[1] for pos in product_positions.values()]
                })

                fig = px.scatter(layout_df, x='X', y='Y', text='Product',
                                 title="Store Layout Map")

                # Add entrance/exit point
                fig.add_trace(go.Scatter(
                    x=[0], y=[0],
                    mode='markers+text',
                    marker=dict(color='red', size=12, symbol='circle'),
                    text=['Entrance/Exit'],
                    textposition="top center",
                    name='Entrance/Exit'
                ))

                # Customize appearance
                fig.update_traces(marker=dict(size=15))
                fig.update_layout(height=500)

                st.plotly_chart(fig, use_container_width=True)
        # Calculate distances
        with styled_container():
            st.markdown("<h3>Transaction Path Analysis :</h3>", unsafe_allow_html=True)

            total_distance = 0
            distances = []
            paths = []

            for _, row in df.iterrows():
                items = [item for item in product_positions if item in df.columns and row.get(item) == 1]

            
                path = [(0, 0)] + [product_positions[item] for item in items] + [(0, 0)]
                

                distance = sum(euclidean(path[i], path[i + 1]) for i in range(len(path) - 1))
                distances.append(distance)
                paths.append(path)
                total_distance += distance

            df['Distance'] = distances

            col1, col2 = st.columns([2, 1])

            with col1:
                if 'transaction_ID'in df.columns:
                 st.dataframe(df[['transaction_ID', 'Distance']].sort_values('Distance', ascending=False),
                             use_container_width=True)
                if 'transaction_ID'not in df.columns:
                    st.dataframe(df[['Distance']].sort_values('Distance', ascending=False),
                             use_container_width=True)

            with col2:
                metrics_cols = st.columns(2)

                with metrics_cols[0]:
                    st.markdown(
                        f"<div style='font-size:13px; font-weight: bold;'>📏 Total Distance<br>"
                        f"<span style='font-size:16px;'>{total_distance:.2f}</span></div>",
                        unsafe_allow_html=True
                    )

                with metrics_cols[1]:
                    st.markdown(
                        f"<div style='font-size:13px; font-weight: bold;'>📏 Average Distance<br>"
                        f"<span style='font-size:16px;'>{df['Distance'].mean():.2f}</span></div>",
                        unsafe_allow_html=True
                    )

        with styled_container():
            st.markdown("<h3>Frequent Itemsets :</h3>", unsafe_allow_html=True)
            if 'transaction_ID'or'Distance'  in df.columns: 
              item_data = df.drop(columns=["transaction_ID", "Distance"], errors='ignore')

            if len(item_data.columns) > 0:
                frequent_itemsets = apriori(item_data.astype(bool),
                                            min_support=min_support,
                                            use_colnames=True)

                col1, col2 = st.columns([2, 1])

                with col1:
                    if len(frequent_itemsets) > 0:
                        # Add itemset size and sort
                        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
                        frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False)

                        # Convert support to percentage
                        frequent_itemsets['support_pct'] = frequent_itemsets['support'] * 100

                        # Convert itemsets to readable string
                        frequent_itemsets['itemsets_str'] = frequent_itemsets['itemsets'].apply(
                            lambda x: ', '.join(sorted(x)))

                        # Display frequent itemsets
                        st.dataframe(
                            frequent_itemsets[['itemsets_str', 'support', 'support_pct', 'length']].head(
                                max_itemsets).style.format({
                                'support': '{:.3f}',
                                'support_pct': '{:.1f}%'
                            }),
                            use_container_width=True
                        )
                    else:
                        st.warning(f"No frequent itemsets found with minimum support of {min_support}")

                with col2:
                    if len(frequent_itemsets) > 0:
                        # Create support distribution chart
                        top_itemsets = frequent_itemsets.head(10).copy()
                        top_itemsets['itemsets_str'] = top_itemsets['itemsets'].apply(lambda x: ', '.join(sorted(x)))

                        support_fig = px.bar(
                            top_itemsets,
                            y='itemsets_str',
                            x='support_pct',
                            title="Top 10 Frequent Itemsets by Support",
                            labels={
                                'support_pct': 'Support (%)',
                                'itemsets_str': 'Itemsets'
                            }
                        )

                        # Format y-axis labels to show the actual items
                        support_fig.update_yaxes(
                            tickmode='array',
                            tickvals=list(range(len(top_itemsets))),
                            ticktext=top_itemsets['itemsets_str'].tolist()
                        )

                        st.plotly_chart(support_fig, use_container_width=True)

        with styled_container():
                #pour afficher la table du les regles
                st.markdown("<h3>Assosiation Rules :</h3>", unsafe_allow_html=True)
                rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
                rules['antecedents']=rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))#pour la affichage et retirer le mot'forzenset'
                rules['consequents']=rules['consequents'].apply(lambda x: ', '.join(sorted(x)))#pour la affichage et retirer le mot'forzenset'
                st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
                # NEW SECTION: Store Layout Optimization with Consolidation
                if len(frequent_itemsets) > 0 and optimize_layout:
                    with styled_container():
                        st.markdown("<h3>Store Layout Optimization with Consolidation :</h3>", unsafe_allow_html=True)

                        # Filter frequent itemsets by minimum size
                        eligible_itemsets = frequent_itemsets[frequent_itemsets['length'] >= 2].copy()

                        if len(eligible_itemsets) > 0:
                            # Optimize product positions based on frequent itemsets
                            consolidated_positions, item_mapping, combined_products = optimize_store_layout(
                                eligible_itemsets, original_product_positions)

                            # Create a before/after comparison
                            col1, col2 = st.columns(2)

                            with col1:
                                st.subheader("Original Layout")

                                # Create original layout visualization
                                orig_layout_df = pd.DataFrame({
                                    'Product': list(original_product_positions.keys()),
                                    'X': [pos[0] for pos in original_product_positions.values()],
                                    'Y': [pos[1] for pos in original_product_positions.values()]
                                })

                                orig_fig = px.scatter(orig_layout_df, x='X', y='Y', text='Product',
                                                      title="Original Store Layout")

                                # Add entrance/exit point
                                orig_fig.add_trace(go.Scatter(
                                    x=[0], y=[0],
                                    mode='markers+text',
                                    marker=dict(color='red', size=12, symbol='circle'),
                                    text=['Entrance/Exit'],
                                    textposition="top center",
                                    name='Entrance/Exit'
                                ))

                                # Customize appearance
                                orig_fig.update_traces(marker=dict(size=15))
                                orig_fig.update_layout(height=400)

                                st.plotly_chart(orig_fig, use_container_width=True)

                            with col2:
                                st.subheader("Consolidated Layout")

                                # Create consolidated layout visualization
                                cons_layout_df = pd.DataFrame({
                                    'Product': list(consolidated_positions.keys()),
                                    'X': [pos[0] for pos in consolidated_positions.values()],
                                    'Y': [pos[1] for pos in consolidated_positions.values()]
                                })

                                # Add tooltip information about consolidated products
                                tooltip_info = []
                                for product in cons_layout_df['Product']:
                                    for name, _, items in combined_products:
                                        if product == name:
                                            tooltip_info.append(f"Contains: {', '.join(sorted(items))}")
                                            break
                                    else:
                                        tooltip_info.append("Individual product")

                                cons_layout_df['Tooltip'] = tooltip_info

                                cons_fig = px.scatter(cons_layout_df, x='X', y='Y', text='Product',
                                                      title="Consolidated Store Layout",
                                                      hover_data=['Tooltip'])

                                # Add entrance/exit point
                                cons_fig.add_trace(go.Scatter(
                                    x=[0], y=[0],
                                    mode='markers+text',
                                    marker=dict(color='red', size=12, symbol='circle'),
                                    text=['Entrance/Exit'],
                                    textposition="top center",
                                    name='Entrance/Exit'
                                ))

                                # Customize appearance
                                cons_fig.update_traces(marker=dict(size=15))
                                cons_fig.update_layout(height=400)

                                st.plotly_chart(cons_fig, use_container_width=True)

                            # Show consolidation details
                            st.subheader("Consolidation Details :")

                            if combined_products:
                                consolidation_df = pd.DataFrame([
                                    {
                                        'Consolidated Product': name,
                                        'Original Products': ', '.join(sorted(items)),
                                        'Position': f"({pos[0]}, {pos[1]})"
                                    }
                                    for name, pos, items in combined_products
                                ])

                                st.dataframe(consolidation_df, use_container_width=True)

                                # Calculate new distances with consolidated layout
                                new_total_distance = 0
                                new_distances = []

                                # Create a reverse mapping for path calculation
                                reverse_mapping = {}
                                for original, consolidated in item_mapping.items():
                                    if consolidated not in reverse_mapping:
                                        reverse_mapping[consolidated] = []
                                    reverse_mapping[consolidated].append(original)

                                for _, row in df.iterrows():
                                    # Find purchased items
                                    purchased_items = [item for item in product_positions if
                                                       item in df.columns and row.get(item) == 1]

                                    # Map them to consolidated products
                                    consolidated_items = set()
                                    for item in purchased_items:
                                        if item in item_mapping:
                                            consolidated_items.add(item_mapping[item])
                                        else:
                                            consolidated_items.add(item)

                                    # Build path with consolidated items
                                    
                                    path = [(0, 0)] + [consolidated_positions[item] for item in
                                                           consolidated_items] + [(0, 0)]
                                    

                                    distance = sum(euclidean(path[i], path[i + 1]) for i in range(len(path) - 1))
                                    new_distances.append(distance)
                                    new_total_distance += distance

                                # Compare distances
                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    st.metric("📏 Original Total Distance", f"{total_distance:.2f}")

                                with col2:
                                    st.metric("📏 Consolidated Total Distance", f"{new_total_distance:.2f}")

                                with col3:
                                    improvement = ((total_distance - new_total_distance) / total_distance) * 100
                                    st.metric("Improvement", f"{improvement:.2f}%")

                        else:
                            st.warning(
                                f"No itemsets found with minimum size .Try lowering the minimum support or itemset size.")

                else:
                 st.error("No product data columns found in the dataset")

    except Exception as e:
        st.error(f"Error processing data: {e}")
else:
    
    st.markdown("""
            <div style="padding: 2rem; border-radius: 8px; border: 1px solid #e0e0e0; height: 100%;">
                <h2 style="margin-bottom: 1rem;">Welcome to Market Basket Analyzer</h2>
                <p style="font-size: 1.1rem;">
                    This tool helps you analyze customer purchasing patterns and optimize your store layout.
                </p>
                <ul style="font-size: 1rem;">
                    <li>Discover which products are frequently purchased together</li>
                    <li>Analyze shopping path distances</li>
                    <li>Optimize product placements for better customer experience</li>
                </ul>
                <p style="margin-top: 1.5rem;">
                    <b>Get started:</b> Upload your dataset using the sidebar on the left
                </p>
            </div>
        """, unsafe_allow_html=True)

    