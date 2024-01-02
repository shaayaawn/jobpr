import streamlit as st
import random
import pandas as pd

class Warehouse:
    def __init__(self, num_products=80, num_orders=50, order_size_range=(1, 5), num_pickers=5, picker_capacity=7):
        self.num_products = num_products
        self.products = [f'p{i}' for i in range(1, num_products + 1)]
        self.layout = self.initialize_layout()
        self.order_size_range = order_size_range
        self.num_orders = num_orders
        self.orders = self.generate_orders()
        self.pickers = [{'id': i + 1, 'capacity': picker_capacity} for i in range(num_pickers)]
        self.dummy_batch = self.dummy_batching()
        self.dummy_routing = self.dummy_routing()
    def initialize_layout(self):
        layout_list = []
        product_choices = self.products.copy()
        random.shuffle(product_choices) 

        for aisle in [1, 3, 5, 7]:
            for y in range(1, 12):  
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
                if current_used_capacity + len(order) <= 7:  # Picker capacity check
                    current_batch.extend(order)
                    current_used_capacity += len(order)
                    i += 1  # Move to the next order
                else:
                    break  # Exit the inner loop to process the next batch

            # Add the current batch to dummy_batch
            dummy_batch.append(current_batch)

            # Update remaining orders by removing the orders processed in this batch
            remaining_orders = remaining_orders[i:]

        return dummy_batch 
    
    #router geht nach findung jedes Produktes zu am nähsten Cross-Aisle und von da direkt nach nächstem Produkt und am Ende zu Startposition
    def dummy_routing(self):
        total_distance = 0
        batch_distance_list = []
        for batch in self.dummy_batching():
            
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
                    
        

def main():
    st.set_page_config(layout="wide")
    warehouse = Warehouse()
    col101,col102,col103 = st.columns([.25,1,.25])
    col102.title("Order batching and picker routing ")   
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    col1,col2,col3 = st.columns([1,.5,1])
    start = col2.button("start an instance")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    if start:
        st.experimental_rerun()
    
    
    
    st.header("Warehaus and batch Information")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    col11,col12 = st.columns(2)
    col11.markdown("### Warehaus Layout")
    col11.dataframe(warehouse.layout)
    col12.markdown("### Orders")
    orders_df = pd.DataFrame({'order_products': warehouse.orders})
    col12.write(orders_df)
    
    col4,col5 = st.columns(2)
    col4.markdown("### Dummy batches")
    dummybatch_df = pd.DataFrame({"batch" : warehouse.dummy_batching()})
    col4.write(dummybatch_df)
    col5.markdown("### Distance for dummy batching and dummy routing")
    
    col5.write(warehouse.dummy_routing)
    col5.write(f'total  ={warehouse.dummy_routing.sum()}' )
    
    # st.write(warehouse.orders)
    # st.write("- -- - -- - - -- - -- - - -")
    # st.write(warehouse.dummy_batching())
    

if __name__ == "__main__":
    main()
