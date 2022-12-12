import os
import configparser
import datetime
import numpy as np
from numpy import average
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from scipy.interpolate.interpnd import CloughTocher2DInterpolator
from jinja2 import Environment, FileSystemLoader
import PyPDF2
from weasyprint import HTML,CSS
import QtUtils
import QtConfigure

def get_num_rows( num_entries, num_cols ):
    return int(num_entries/num_cols)

def grid(x, y, z, resX=100, resY=100):

    xi = np.linspace(min(x), max(x), resX)
    yi = np.linspace(min(y), max(y), resY)
    X, Y = np.meshgrid(xi, yi)
    Z =  griddata((x, y) , z ,  (xi[None,:], yi[:,None]) ,method='cubic'   )
    
    return X, Y, Z

class QtReport:

    def __init__(self , df , * , report_file_name , agent_name , airport_name , way_name ):
        
        #Configure the DLL searc path the weasyprint module depends on 
        QtUtils.setDLLSearchPath()

        #Configure the underlying settings
        QtConfigure.QtConfig()

        self.config = configparser.ConfigParser();
        self.config.read("QtConfig.ini")
        
        self.reportFileName = report_file_name
        self.agentName = agent_name
        self.airportName = airport_name
        self.wayName = way_name
        
        #Initialze a list for keeping track of the individual pdf files generated 
        self.pdfNames = []
        #Store dataframe
        self.df = df
        #Transform dataframe
        self.__transformDF()

        self.gminI = self.df['I'].min()
        self.gmaxI = self.df['I'].max()

        return

    def __transformDF(self):
        #Extract all the light sources and sort them in increasing order of their ids
        id_list = np.sort(self.df['Light ID'].unique())
    
        imax_list = []
        iavg_list = []
        hmax_list = []
        vmax_list = []
        icao_list = []
        color_list = []
        outcome_list = []

        #Extract necessary information from the initial dataframe
        for id in id_list:

            imax = self.df.loc[self.df['Light ID'] == id]['I'].max()
            maxIDF = self.df.loc[self.df['I'] == imax]
            hmax = maxIDF.iloc[0]['H']
            vmax = maxIDF.iloc[0]['V']
            iavg = self.df.loc[self.df['Light ID'] == id]['I'].mean()

            icao = maxIDF.iloc[0]['%ICAO']
            color = maxIDF.iloc[0]['C']
            outcome = maxIDF.iloc[0]['S']

            imax_list.append(imax)
            iavg_list.append(iavg)
            hmax_list.append(hmax)
            vmax_list.append(vmax)
            icao_list.append(icao)
            color_list.append(color)
            outcome_list.append(outcome)

        data = {
                'Light ID': id_list,
                '%ICAO': icao_list,
                'AVG(cd)': iavg_list,
                'Max(cd)' : imax_list,
                'Hmax(Deg)' : hmax_list,
                'Vmax(Deg)' : vmax_list,
                'C' : color_list,
                'S' : outcome_list
            }
        
        columns =('Light ID','%ICAO', 'AVG(cd)','Max(cd)','Hmax(Deg)','Vmax(Deg)','C','S')

        #construct a dataframe 
        self.mtab_df = pd.DataFrame(data,columns=columns)
        
        #set 'LightID' as index
        self.mtab_df.set_index(['Light ID'],inplace=True)

        return
    
    def __barPlot( self , cur_df, page_no , start_row , end_row ):
        light_ids = list(range(start_row+1,end_row+2))

        #print('Bar Plot : {0}--->{1}'.format(start_row+1,end_row+1))

        average_values = []
        ICAOs = []

        for _ , row in cur_df.iterrows():
            average_values.append(row['AVG(cd)'])
            ICAOs.append(row['%ICAO'])

        colors = []

        for _ , row in cur_df.iterrows():
            if row['C'] == 'R':
                colors.append('red')
            elif row['C'] == 'Y':
                colors.append('orange')
            elif row['C'] == 'W':
                colors.append('yellow')
            elif row['C'] == 'G':
                colors.append('green')
            else:
                colors.append('grey')

        #List light ids as x-ticks on the x-axis
        xticks = range(light_ids[0],light_ids[-1]+1,2)

        diff = self.num_rows_per_page - len(light_ids)

        #Fill in hiden dummy values
        if diff > 0:
            for i in range(light_ids[-1]+1,light_ids[-1]+diff+1):
                light_ids.append(i)
                average_values.append(0.0)
                ICAOs.append(0.0)

        red_values = []
        edge_colors = []
        line_widths = []

        for row , val in enumerate(ICAOs):
            if row < cur_df.shape[0]:
                if val < 50.0:
                    red_values.append(average_values[row]+500)
                    edge_colors.append('red')
                    line_widths.append(1.0)
                else:
                    red_values.append(0.0)
                    edge_colors.append('white')
                    line_widths.append(0.0)
            else:
                red_values.append(0.0)  
                edge_colors.append('white')
                line_widths.append(0.0)
           
        #Set outer background color
        plt.figure(facecolor='#9d9d9d')
        #Set inner background color
        plt.axes().set_facecolor('#9d9d9d')

        #Draw a horizontal grid
        plt.gca().yaxis.grid()
        plt.gca().set_axisbelow(True)

        plt.xticks(xticks)
        plt.xlabel('Light ID')
        plt.ylabel('Average Candela (Cd)')

        width = float(self.config['BarChartFormat']['widthofonebar'])
        
        plt.bar( x=light_ids , height=red_values  , width=width, color='#9d9d9d' , edgecolor=edge_colors , linewidth=line_widths )
        plt.bar( x=light_ids , height=average_values , width=width*0.50, color=colors  )

        save_as = os.path.join( self.config['Locations']['templocation'] , '{0}-{1}-Bars.png'.format(self.reportFileName,page_no))

        #save the plot
        plt.savefig( save_as , dpi=400  )
        plt.close()
        return

    def __contourPlot( self , page_no , start_row , end_row ):

        H = []
        V = []
        I = []
        Hmax = []
        Vmax = []
        
        #Prepare the specified number of sets of data to be plotted (default=6)
        for cur_id in range(start_row+1 , end_row+2): #starr_row+1--->end_row+1

            cur_df = self.df.loc[self.df['Light ID'] == cur_id]

            imax = cur_df['I'].max()
            maxIDF = self.df.loc[self.df['I'] == imax]
            hmax = maxIDF.iloc[0]['H']
            vmax = maxIDF.iloc[0]['V']

            #print('Light ID {} Hmax {} Vmax {}'.format(cur_id,hmax,vmax))

            cur_idx = (cur_id-1)%self.num_rows_per_page

            H.append(cur_df['H'])
            V.append(cur_df['V'])
            I.append(cur_df['I'])
            Hmax.append(hmax)
            Vmax.append(vmax)

            H[cur_idx],V[cur_idx],I[cur_idx] = grid(H[cur_idx],V[cur_idx],I[cur_idx])

        #Compute the number of subplots
        nSubplots = len(H)

        #Set up plots
        nrows = self.num_rows_per_page

        fig, ax = plt.subplots(nrows=1, ncols=nrows,figsize=(50,6))
        plt.subplots_adjust(wspace=0.01,left=0.025, right=1.0)

        #Choose the number of contour levels
        nlevels = int(self.config['ContourFormat']['nlevels'])
        levels = np.linspace(0,self.gmaxI,nlevels+1)
        cticks = np.arange(0,self.gmaxI,2000)

        #Determine whether to disable the axis for each (row,col)
        for row in range(0,nrows):
                if row < nSubplots:
                    ax[row].axis('on')
                else:
                    ax[row].axis('off')

        #plot the contour for each entry
        for row in range(0,nrows):
            if row <nSubplots:   
                cs = ax[row].contour(H[row],V[row],I[row], levels=levels, linewidths=0.4, linestyles='dashed', colors='k') 
                    
                csf = ax[row].contourf(H[row],V[row],I[row],levels=levels, cmap='Spectral_r',extend='both')
                    
                ax[row].plot([Hmax[row]],[Vmax[row]],marker='X',color='black')

                ax[row].set_title('Light ID: {0}'.format(start_row+row+1),fontsize=10)

        fig.supxlabel('Horizontal Degrees')
        fig.supylabel('Vertical Degrees')
        fig.suptitle('Vertical Scanning')

        fig.subplots_adjust(right=0.8)
        bar_ax = fig.add_axes([0.82, 0.15, 0.001, 0.80])
        fig.colorbar(csf,cax=bar_ax,ticks=cticks) 
        #set up the title and the x- and y-axis
        

        #save the plot as an image file
        save_as = os.path.join( self.config['Locations']['templocation'] , '{0}-{1}-Contours.png'.format(self.reportFileName,page_no))

        #save the plot
        plt.savefig( save_as , dpi=400 ,bbox_inches='tight' )
        plt.close()
        
        return

    def __onePDF( self , * , html_page , page_no ):

        save_as = os.path.join( self.config['Locations']['templocation'] , '{0}-{1}.pdf'.format(self.reportFileName,page_no) )

        #Set base url to img folder
        HTML( string=html_page , base_url='img' ).write_pdf(save_as) 

        QtUtils.displayInfo('{0} was made...'.format(save_as))

        return
    
    def __generateOnePDF( self , page_no , start_row , end_row ):

         #Draw contours
        self.__contourPlot(page_no, start_row, end_row)

        #Get the entries for the current page
        cur_df = self.mtab_df.iloc[start_row:end_row+1] #end_row exclusive

        #print('GenerateOnePDF : Page {0} --> {1}---{2}.pdf'.format(page_no,start_row+1,end_row+1))

        self.__barPlot( cur_df, page_no , start_row , end_row )

        #Convert the dataframe into an HTML table, excluding the index column
        m_table = cur_df.to_html(index=True) 

        #datetime_of_report = datetime.datetime.today()

         #Render each page 
        html_page =  self.template.render(
                                    m_table=m_table,
                                    page_no=page_no, 
                                    report_file_name=self.reportFileName,  
                                    air_port_name=self.airportName,
                                    way_name=self.wayName,
                                    agent_name=self.agentName,  
                                    date_of_report=QtUtils.getDate(self.datetime_of_report),
                                    time_of_report=QtUtils.getTime(self.datetime_of_report),
                                    caption_path='caption.png',
                                    bar_plot_path='{0}-{1}-Bars.png'.format(self.reportFileName,page_no),
                                    contour_plot_path='{0}-{1}-Contours.png'.format(self.reportFileName,page_no)   
                                )

        self.pdfNames.append('{0}-{1}.pdf'.format(self.reportFileName,page_no))

        #Compute the name of the current HTML
        save_as = os.path.join( self.config['Locations']['templocation'] , '{0}-{1}.html'.format(self.reportFileName,page_no) ) 
        
        with open( save_as , 'w' , encoding='utf-8') as html_file: 
            html_file.write(html_page)

        QtUtils.displayInfo('{0} was made...'.format(save_as))

        #Export as a pdf file
        self.__onePDF( html_page=html_page , page_no=page_no )

        return

    def __mergePDFs(self):

        input_dir = self.config['Locations']['templocation']
        output_dir = self.config['Locations']['reportlocation']

        merge_list = []

        for f in os.listdir(input_dir):
            if f in self.pdfNames:
                merge_list.append(os.path.join(input_dir,f))

        sorted(merge_list)

        merger = PyPDF2.PdfFileMerger()

        for f in merge_list:
            merger.append(f)

        save_as = os.path.join(output_dir,'{0}.pdf'.format(self.reportFileName))
        merger.write(save_as) 
        merger.close()

        return
    
    def generate( self ):

        file_loader = FileSystemLoader(self.config['Locations']['templatelocation']) 
        env = Environment(loader=file_loader,trim_blocks=True)
        self.template = env.get_template('template.html') 

        #Get date and time
        self.datetime_of_report = datetime.datetime.today()

        #Get the total number of entries
        num_of_rows  = self.mtab_df.shape[0]
    
        #Get the number of rows per page
        self.num_rows_per_page = int(self.config['ReportFormat']['numberofrowsperpage'])
        
        #Calculate the number of pages based on the config where the number of entries per page is set
        num_of_pages = int(np.ceil(num_of_rows / self.num_rows_per_page))

        row = 0
        for page_no in range( 1 , num_of_pages+1 ):
            start_row = row
            end_row = start_row + self.num_rows_per_page -1

            if end_row > num_of_rows-1:
                end_row = num_of_rows-1

            #Export the current page 
            self.__generateOnePDF( page_no , start_row , end_row )

            row = end_row+1

        #Merge PDFs
        self.__mergePDFs()

        #Delete temp files
        dir = self.config['Locations']['templocation']
        for f in os.listdir(dir):
          os.remove(os.path.join(dir,f))

        return