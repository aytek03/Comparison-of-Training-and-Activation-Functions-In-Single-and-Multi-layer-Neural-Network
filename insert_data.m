function [ sinif, data ] = insert_data()


clc
clear all
load wdbcorg.data;
data=wdbcorg(:,3:32);
for i=1:569
if wdbcorg(i,2)==0 
sinif{i,1}='M';
else
sinif{i,1}='B';
end
end