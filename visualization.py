import pandas as pd
import plotly.express as px

class DataVisualization:
    def __init__(self):
        try:
            self.df = pd.read_csv("2019.csv")
        except Exception as e:
            print(f"Error loading data: {e}")
            self.df = None

    def create_choropleth(self):
        """Create world happiness score map"""
        if self.df is None:
            return None

        fig = px.choropleth(
            self.df,
            locations="Country or region",
            locationmode="country names",
            color="Score",
            hover_name="Country or region",
            color_continuous_scale="greens",
            title="World Happiness Score by Country"
        )
        return fig.to_html(full_html=False)

    def create_bar_chart(self):
        """Create bar chart of top 10 happiest countries"""
        if self.df is None:
            return None

        fig = px.bar(
            self.df,
            x="Country or region",
            y="Score",
            text="Score",
            color="Score",
            color_continuous_scale="viridis",
            title="Top 10 Happiest Countries",
            labels={"Score": "Happiness Score"}
        )
        fig.update_layout(xaxis_tickangle=-45)
        return fig.to_html(full_html=False)

    def create_scatter_plot(self):
        """Create GDP vs Happiness scatter plot"""
        if self.df is None:
            return None

        fig = px.scatter(
            self.df,
            x="GDP per capita",
            y="Score",
            text="Country or region",
            size="Score",
            color="Score",
            color_continuous_scale="magma",
            title="GDP vs Happiness Score",
            labels={"Score": "Happiness Score", "GDP per capita": "GDP per Capita"}
        )
        return fig.to_html(full_html=False)

    def create_animated_scatter_plot(self):
        """Create animated scatter plot for happiness evolution over years"""
        if self.df is None:
            return None

        df_new = self.df.copy()
        df_new["Year"] = 2020

        fig = px.scatter(
            df_new,
            x="GDP per capita",
            y="Score",
            size="Score",
            color="Country or region",
            animation_frame="Year",
            title="Happiness Evolution Over Years"
        )
        return fig.to_html(full_html=False)

    def get_all_plots(self):
        """Generate all plots and return as HTML"""
        try:
            plots = {
                'choropleth': self.create_choropleth() if self.df is not None else None,
                'bar_chart': self.create_bar_chart() if self.df is not None else None,
                'scatter_plot': self.create_scatter_plot() if self.df is not None else None,
                'animated_scatter_plot': self.create_animated_scatter_plot() if self.df is not None else None
            }
            return plots
        except Exception as e:
            print(f"Error creating plots: {e}")
            return None

