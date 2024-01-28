import streamlit as st
import random
import pandas as pd

class Warehouse:
    def __init__(self, num_products=80, num_orders=50, order_size_range=(1, 10), num_pickers=5, picker_capacity=10):
        self.num_products = num_products
        self.picker_capacity = picker_capacity
        self.products = [f'p{i}' for i in range(1, num_products + 1)]
        self.layout = self.initialize_layout()
        self.order_size_range = order_size_range
        self.num_orders = num_orders
        self.orders = self.generate_orders()
        self.pickers = [{'id': i + 1, 'capacity': picker_capacity} for i in range(num_pickers)]
        self.dummy_batch = self.dummy_batching()
        # self.dummy_routing = self.dummy_routing()
    def initialize_layout(self):
        layout_list = []
        product_choices = self.products.copy()
        random.shuffle(product_choices) 

        for aisle in [1, 4, 7, 10]:
            for y in range(1, 12): 
                if y == 6:
                    continue
                for side in ['L', 'R']:  
                    product = product_choices.pop() if product_choices else None
                    layout_list.append({'Aisle': aisle, 'Position': y, 'Side': side, 'Product': product})

        layout_df = pd.DataFrame(layout_list)
        layout_df.set_index("Product",inplace=True)
        layout_df = layout_df.sort_values(by="Product")
        return layout_df

    
    
    def generate_orders(self):
        orders = []
        for _ in range(self.num_orders):
            order_size = random.randint(*self.order_size_range)  
            order = random.sample(self.products, order_size)
            orders.append(order)
        return orders
    
    def dummy_batching(self):
        remaining_orders = self.orders.copy()
        dummy_batch = []
        while remaining_orders:
            current_batch = []
            current_used_capacity = 0

            # We'll use a separate loop index because we need to modify remaining_orders within the loop
            i = 0  
            while i < len(remaining_orders):
                order = remaining_orders[i]
                if current_used_capacity + len(order) <= self.picker_capacity:  # Picker capacity check
                    current_batch.extend(order)
                    current_used_capacity += len(order)
                    i += 1  # Move to the next order
                else:
                    break  # Exit the inner loop to process the next batch

            # Add the current batch to dummy_batch
            dummy_batch.append(list(set(current_batch)))

            # Update remaining orders by removing the orders processed in this batch
            remaining_orders = remaining_orders[i:]

        return dummy_batch 
    
    #router geht nach findung jedes Produktes zu am nähsten Cross-Aisle und von da direkt nach nächstem Produkt und am Ende zu Startposition
    def dummy_routing(self,batching_algorithm_output):
        
        total_distance = 0
        batch_distance_list = []
        for batch in batching_algorithm_output:
            
            current_location = (0,0)
            batch_distance = 0
            for item in batch:
                
                item_location = (self.layout.loc[item,"Aisle"],self.layout.loc[item,"Position"])
                distance = abs(item_location[0]-current_location[0]) + abs(item_location[1]-current_location[1])
                total_distance+=distance
                batch_distance += distance
                if item_location[1]<=2:
                    current_location = (item_location[0],0)
                    total_distance += abs(item_location[1])
                    batch_distance +=  abs(item_location[1])
                elif ((3<= item_location[1])&(item_location[1])<=9):
                    current_location = (item_location[0],6)
                    total_distance += abs(item_location[1]-6)
                    batch_distance += abs(item_location[1]-6)
                else:
                    current_location = (item_location[0],12)
                    total_distance += abs(item_location[1]-12)
                    batch_distance += abs(item_location[1]-12)
            total_distance += item_location[0] + item_location[1]
            batch_distance += item_location[0] + item_location[1]
            batch_distance_list.append(batch_distance)
        df = pd.DataFrame({"batch_distance" : batch_distance_list})
            
        return df
                    
    def time_saving_batching(self):
        #es kann sein dass mit diesem Algorithmus nur 2 beste Orders gebatched werden und da bleibt viel Platz noch für Picker übrig dass sonst von andere Orders benutzt werden könnte
        orders = self.orders.copy()
        saved_distance_list = []
        for i in range(len(self.orders)):
            for j in range(i+1,len(self.orders)):
                first_order_distance= self.calculate_order_distance(orders[i])
                second_order_distance= self.calculate_order_distance(orders[j])
                combined_order_set = list(set(orders[i]+orders[j]))
                combined_order_distance= self.calculate_order_distance(combined_order_set)
                saved_distance = first_order_distance + second_order_distance - combined_order_distance
                saved_distance_list.append(((i, j), saved_distance))
        saved_distance_list.sort(key=lambda x: x[1], reverse=True)
        
        proceeded_orders = set()
        batch_index_list = []
        for (i,j),_ in saved_distance_list:
            if i in proceeded_orders or j in proceeded_orders:
                continue
            if len(orders[i]) + len(orders[j]) <= self.picker_capacity:
                
                batch_index_list.append([i,j])
                proceeded_orders.update([i,j])
        remaining_orders = [i for i in range(len(orders)) if i not in proceeded_orders]
        for i in remaining_orders:
            
            batch_index_list.append([i])
        final_batch_list = []
        for batch_indices in batch_index_list:
            combined_products = []
            for index in batch_indices:
                combined_products.extend(orders[index])
            unique_products = set(combined_products)
            final_batch_list.append(list(unique_products))
           
        return final_batch_list
    
    
    
    def s_shape_routing(self,batching_algorithm_output):
        
        total_distance = 0
        batch_distance_list = []
    
        # List to store batch index, aisle, and block information
        batch_aisle_block_info = []
    
        for batch_index, batch in enumerate(batching_algorithm_output):
            for p in batch:
                # Get aisle and position for the product
                aisle = self.layout.loc[p, "Aisle"]
                position = self.layout.loc[p, "Position"]
    
                # Determine the block based on position
                block = 1 if position < 6 else 2
    
                # Create a tuple of batch index, aisle, and block
                batch_aisle_block_tuple = (batch_index, aisle, block)     
                batch_aisle_block_info.append(batch_aisle_block_tuple)
        
        for batch_index in range(len(batching_algorithm_output)):
        
            aisles_in_batch = [(aisle, block) for _, aisle, block in batch_aisle_block_info if _ == batch_index]

       
            aisles_block_1 = sorted(set([aisle for aisle, block in aisles_in_batch if block == 1]))
            aisles_block_2 = sorted(set([aisle for aisle, block in aisles_in_batch if block == 2]), reverse=True)

            current_location = (0, 0)
            batch_distance = 0
            
            # Process Block 1
            if aisles_block_1 and aisles_block_2:
                distance_block_1 = aisles_block_1[-1] + 5*len(aisles_block_1)
                if len(aisles_block_1)%2 ==1:
                    current_location = (aisles_block_1[-1],6)
                else:
                    current_location = (aisles_block_1[-1],0)
                distance_block_2 = abs(current_location[0] - aisles_block_2[-1]) + 5*len(aisles_block_2)
                batch_distance= distance_block_1+distance_block_2+abs(aisles_block_2[-1]+9)
                batch_distance_list.append(batch_distance)
    
            # Process Block 2
            elif aisles_block_1:
                distance_block_1 = 2*aisles_block_1[-1] + 5*len(aisles_block_1)
                if len(aisles_block_1)%2 ==1:
                    batch_distance = distance_block_1 + 5
                else:
                    batch_distance=distance_block_1
                batch_distance_list.append(batch_distance)
                
                
            elif aisles_block_2:
                distance_block_2 = 2*aisles_block_2[0] + 5*len(aisles_block_2)
                if len(aisles_block_2)%2 ==1:
                    batch_distance +=11
                else:
                    batch_distance +=6
                batch_distance_list.append(batch_distance)
                
            df = pd.DataFrame({"batch_distance" : batch_distance_list})

        return  df
        
        
      
    

        
    
    
    
    
    
    
    def calculate_order_distance(self,order):
        
        #hier sollen in besten fall algorithmus so ändern das es die Producte in Beste Reihefolge sortiert so dass minimum distance berechnet wird
        start_order = order[0]
        current_location = (self.layout.loc[start_order,"Aisle"],self.layout.loc[start_order,"Position"])
        distance=0
        for product in order:
            product_location = (self.layout.loc[product,"Aisle"],self.layout.loc[product,"Position"])
            distance+= abs(product_location[0]-current_location[0])+ abs(product_location[1]-current_location[1])
            current_location = product_location
        distance += current_location[0]+current_location[1]
        return distance
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

def main():
    # Set page configuration
    st.set_page_config(layout="wide", page_title="Warehouse Order Batching and Routing")
    warehouse = Warehouse()

    # Page title
    st.title("Order Batching and Picker Routing")

    # Start instance button
    if st.button("Start an Instance"):
        st.experimental_rerun()

    # Warehouse and Batch Information Header
    st.header("Warehouse and Batch Information")

    # Display Warehouse Layout and Orders in two columns
    col_layout, col_orders = st.columns(2)
    with col_layout:
        st.subheader("Warehouse Layout")
        st.dataframe(warehouse.layout)

    with col_orders:
        st.subheader("Orders")
        orders_df = pd.DataFrame({'order_products': warehouse.orders})
        st.dataframe(orders_df)

    # Dummy Batches and Time Saving Batches Analysis
    st.subheader("Batching and Routing Analysis")
    
    # Dummy Batches
    dummy_batches = warehouse.dummy_batching()
    col_dummy_batch, col_dummy_route1, col_dummy_route2 = st.columns(3)
    with col_dummy_batch:
        st.markdown("#### Custom Batches")
        st.dataframe(pd.DataFrame({"batch": dummy_batches}))

    with col_dummy_route1:
        st.markdown("#### Distance for Custom Batching & Custom Routing")
        dummy_route_distance = warehouse.dummy_routing(dummy_batches)
        st.dataframe(dummy_route_distance)
        st.metric("Total Distance", dummy_route_distance.sum())

    with col_dummy_route2:
        st.markdown("#### Distance for Custom Batching & Time Saving Routing")
        time_saving_route_distance = warehouse.s_shape_routing(dummy_batches)
        st.dataframe(time_saving_route_distance)
        st.metric("Total Distance", time_saving_route_distance.sum())

    # Time Saving Batches
    time_saving_batches = warehouse.time_saving_batching()
    col_time_saving_batch, col_time_saving_route1, col_time_saving_route2 = st.columns(3)
    with col_time_saving_batch:
        st.markdown("#### Time Saving Batches")
        st.dataframe(pd.DataFrame({"batch": time_saving_batches}))

    with col_time_saving_route1:
        st.markdown("#### Distance for Time Saving Batching & Custom Routing")
        time_saving_dummy_route_distance = warehouse.dummy_routing(time_saving_batches)
        st.dataframe(time_saving_dummy_route_distance)
        st.metric("Total Distance", time_saving_dummy_route_distance.sum())

    with col_time_saving_route2:
        st.markdown("#### Distance for Time Saving Batching & S Shape Routing")
        time_saving_s_shape_route_distance = warehouse.s_shape_routing(time_saving_batches)
        st.dataframe(time_saving_s_shape_route_distance)
        st.metric("Total Distance", time_saving_s_shape_route_distance.sum())

if __name__ == "__main__":
    main()
